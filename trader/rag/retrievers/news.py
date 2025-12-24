"""
NewsRetriever: Retrieve news_piece candidates
"""
import json
import sqlite3
from typing import List
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.rag.types import RetrievalPlan, Candidate, Document
from trader.rag.db.queries import get_db_connection
from trader.news.prepare import parse_news_json, clean_html, extract_text_from_news
from trader.news.analyze import analyze
from trader.logger import get_logger

logger = get_logger(__name__)


class NewsRetriever:
    """News retriever"""
    
    def retrieve(self, plan: RetrievalPlan) -> List[Candidate]:
        """
        Retrieve news candidates
        
        Args:
            plan: Retrieval plan
            
        Returns:
            List[Candidate]
        """
        need = plan.needs.get("news_piece")
        if not need or not need.enable:
            logger.debug("News retrieval not enabled")
            return []
        
        conn = get_db_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        candidates = []
        
        try:
            # Build query conditions
            conditions = []
            params = []
            
            if plan.stock_code:
                conditions.append("stock_code = ?")
                params.append(plan.stock_code)
            
            # Time window filter (using datetime field)
            conditions.append("datetime BETWEEN ? AND ?")
            params.extend([plan.time_start[:10], plan.time_end[:10]])  # Only date part
            
            # Only retrieve records with news
            conditions.append("news IS NOT NULL AND news != ''")
            
            # Impact threshold filter (if in constraints)
            min_impact = plan.constraints.get("min_impact", 0)
            if min_impact > 0:
                # Note: raw_data table may not have direct impact field
                # Skip filtering here, handle in rerank stage
                pass
            
            where_clause = " AND ".join(conditions)
            
            # Check if analysis result fields exist
            cursor.execute("PRAGMA table_info(raw_data)")
            columns = [col[1] for col in cursor.fetchall()]
            has_sentiment = 'news_sentiment_mean' in columns
            has_impact = 'news_impact_mean' in columns
            
            # Build SELECT fields
            select_fields = ["id", "stock_code", "datetime", "news"]
            if has_sentiment:
                select_fields.append("news_sentiment_mean")
            if has_impact:
                select_fields.append("news_impact_mean")
            
            # Query SQL
            query = f"""
                SELECT {', '.join(select_fields)}
                FROM raw_data
                WHERE {where_clause}
                ORDER BY datetime DESC
                LIMIT ?
            """
            params.append(need.recall_k)
            
            logger.debug(f"Executing news retrieval query: {query[:200]}...")
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            logger.info(f"News retrieval returned {len(rows)} records")
            
            # Convert to Document and Candidate
            for row in rows:
                try:
                    doc = self._row_to_document(row, has_sentiment, has_impact)
                    if doc:
                        # Use simple rule score (based on time decay and analysis results)
                        score = self._calculate_recall_score(row, plan)
                        candidate = Candidate(
                            doc=doc,
                            recall_score=score,
                            recall_source="sql_filter"
                        )
                        candidates.append(candidate)
                except Exception as e:
                    logger.warning(f"Failed to convert news record (id={row['id']}): {e}")
                    continue
            
        except Exception as e:
            logger.error(f"News retrieval failed: {e}", exc_info=True)
        finally:
            conn.close()
        
        logger.info(f"News retrieval completed, returned {len(candidates)} candidates")
        return candidates
    
    def _row_to_document(self, row: sqlite3.Row, has_sentiment: bool = False, has_impact: bool = False) -> Document:
        """Convert database row to Document"""
        raw_data_id = row['id']
        stock_code = row['stock_code']
        datetime_str = row['datetime']
        news_raw = row['news']
        
        # Parse news JSON
        news_obj = parse_news_json(news_raw)
        if not news_obj:
            return None
        
        # Get publish time (prefer publish_time from news, otherwise use datetime)
        publish_time = news_obj.get('publish_time', datetime_str)
        
        # Extract text content
        text = extract_text_from_news(news_raw)
        
        # Build payload
        payload = {
            "title": news_obj.get('title', ''),
            "content": news_obj.get('content', ''),
            "publish_time": publish_time,
            "raw_data_id": raw_data_id
        }
        
        # Try to get analysis results from database row (if exists)
        if has_sentiment and row.get('news_sentiment_mean') is not None:
            payload["sentiment"] = row['news_sentiment_mean']
        if has_impact and row.get('news_impact_mean') is not None:
            payload["impact"] = row['news_impact_mean']
        
        doc = Document(
            doc_id=f"news:{raw_data_id}:{publish_time}",
            doc_type="news_piece",
            stock_code=stock_code,
            timestamp=publish_time,
            payload=payload,
            text=text
        )
        
        return doc
    
    def _calculate_recall_score(self, row: sqlite3.Row, plan: RetrievalPlan) -> float:
        """Calculate recall score"""
        # Simple rule: based on time decay
        # News closer to decision_time gets higher score
        try:
            from datetime import datetime
            row_time = datetime.fromisoformat(row['datetime'].replace('Z', '+00:00'))
            plan_time = datetime.fromisoformat(plan.time_end.replace('Z', '+00:00'))
            days_diff = abs((plan_time - row_time).days)
            
            # Time decay: exp(-days/7), higher weight within 7 days
            import math
            time_score = math.exp(-days_diff / 7.0)
            
            # Base score
            base_score = 0.5
            
            # Bonus if has news content
            if row['news']:
                base_score += 0.3
            
            return base_score * time_score
        except Exception:
            return 0.5  # Default score

