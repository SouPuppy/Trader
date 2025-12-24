"""
TrendsRetriever: Retrieve trend feature candidates (feature similarity)
"""
import sqlite3
from typing import List, Optional
import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.rag.types import RetrievalPlan, Candidate, Document
from trader.rag.db.queries import get_db_connection
from trader.logger import get_logger

logger = get_logger(__name__)


class TrendsRetriever:
    """Trend feature retriever"""
    
    def retrieve(self, plan: RetrievalPlan) -> List[Candidate]:
        """
        Retrieve trend feature candidates
        
        Args:
            plan: Retrieval plan
            
        Returns:
            List[Candidate]
        """
        need = plan.needs.get("trends")
        if not need or not need.enable:
            logger.debug("Trend feature retrieval not enabled")
            return []
        
        conn = get_db_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        candidates = []
        
        try:
            # Get current feature vector (for similarity calculation)
            current_features = self._get_current_features(plan, cursor)
            
            # Build query conditions
            conditions = []
            params = []
            
            if plan.stock_code:
                conditions.append("stock_code = ?")
                params.append(plan.stock_code)
            
            # Time window filter (features table uses datetime column)
            conditions.append("datetime BETWEEN ? AND ?")
            params.extend([plan.time_start, plan.time_end])
            
            where_clause = " AND ".join(conditions)
            
            # Query SQL (get all candidates first, then calculate similarity)
            # Using features table directly with datetime column
            query = f"""
                SELECT stock_code, datetime,
                       ret_1d, ret_5d, ret_20d, range_pct, gap_pct, close_to_open,
                       vol_20d, vol_60d, vol_z_20d,
                       pe_ratio, pe_ratio_ttm, pcf_ratio_ttm, pb_ratio, ps_ratio, ps_ratio_ttm
                FROM features
                WHERE {where_clause}
                ORDER BY datetime DESC
                LIMIT ?
            """
            params.append(need.recall_k * 2)  # Get more, then sort by similarity
            
            logger.debug(f"Executing trend feature retrieval query: {query[:200]}...")
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            logger.info(f"Trend feature retrieval returned {len(rows)} records")
            
            # Convert to Document and Candidate, and calculate similarity
            candidate_list = []
            for row in rows:
                try:
                    doc = self._row_to_document(row)
                    if doc:
                        # Calculate similarity score
                        score = self._calculate_similarity_score(row, current_features, plan)
                        candidate = Candidate(
                            doc=doc,
                            recall_score=score,
                            recall_source="feature_knn"
                        )
                        candidate_list.append(candidate)
                except Exception as e:
                    try:
                        stock_code_val = row['stock_code']
                        datetime_val = row['datetime']
                        logger.warning(f"Failed to convert trend record (stock_code={stock_code_val}, datetime={datetime_val}): {e}")
                    except:
                        logger.warning(f"Failed to convert trend record: {e}")
                    continue
            
            # Sort by similarity score, take top-k
            candidate_list.sort(key=lambda c: c.recall_score, reverse=True)
            candidates = candidate_list[:need.recall_k]
            
        except Exception as e:
            logger.error(f"Trend feature retrieval failed: {e}", exc_info=True)
        finally:
            conn.close()
        
        logger.info(f"Trend feature retrieval completed, returned {len(candidates)} candidates")
        return candidates
    
    def _get_current_features(self, plan: RetrievalPlan, cursor: sqlite3.Cursor) -> Optional[dict]:
        """Get current feature vector"""
        try:
            query = """
                SELECT ret_1d, ret_5d, ret_20d, range_pct, gap_pct, close_to_open,
                       vol_20d, vol_60d, vol_z_20d,
                       pe_ratio, pe_ratio_ttm, pcf_ratio_ttm, pb_ratio, ps_ratio, ps_ratio_ttm
                FROM features
                WHERE stock_code = ?
                ORDER BY datetime DESC
                LIMIT 1
            """
            params = [plan.stock_code]
            
            cursor.execute(query, params)
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
        except Exception:
            return None
    
    def _row_to_document(self, row: sqlite3.Row) -> Document:
        """Convert database row to Document"""
        stock_code = row['stock_code']
        datetime_str = row['datetime']
        # Features table stores daily data, so frequency is always '1d'
        frequency = '1d'
        
        # Extract all feature fields
        # sqlite3.Row doesn't support .get(), use dictionary access directly
        # NULL values in SQLite will be returned as None in Python
        payload = {
            "frequency": frequency,
            "ret_1d": row['ret_1d'],
            "ret_5d": row['ret_5d'],
            "ret_20d": row['ret_20d'],
            "range_pct": row['range_pct'],
            "gap_pct": row['gap_pct'],
            "close_to_open": row['close_to_open'],
            "vol_20d": row['vol_20d'],
            "vol_60d": row['vol_60d'],
            "vol_z_20d": row['vol_z_20d'],
            "pe_ratio": row['pe_ratio'],
            "pe_ratio_ttm": row['pe_ratio_ttm'],
            "pcf_ratio_ttm": row['pcf_ratio_ttm'],
            "pb_ratio": row['pb_ratio'],
            "ps_ratio": row['ps_ratio'],
            "ps_ratio_ttm": row['ps_ratio_ttm'],
        }
        
        # Use stock_code and datetime as doc_id (features table uses (stock_code, datetime) as primary key)
        doc = Document(
            doc_id=f"trends:{stock_code}:{datetime_str}",
            doc_type="trends",
            stock_code=stock_code,
            timestamp=datetime_str,
            payload=payload,
            text=None
        )
        
        return doc
    
    def _calculate_similarity_score(
        self, 
        row: sqlite3.Row, 
        current_features: Optional[dict],
        plan: RetrievalPlan
    ) -> float:
        """Calculate similarity score"""
        # If no current features, use time decay
        if not current_features:
            return self._time_decay_score(row, plan)
        
        # Extract feature vectors
        feature_names = [
            'ret_1d', 'ret_5d', 'ret_20d', 'range_pct', 'gap_pct', 'close_to_open',
            'vol_20d', 'vol_60d', 'vol_z_20d',
            'pe_ratio', 'pe_ratio_ttm', 'pcf_ratio_ttm', 'pb_ratio', 'ps_ratio', 'ps_ratio_ttm'
        ]
        
        try:
            # Build feature vectors
            current_vec = []
            row_vec = []
            
            for name in feature_names:
                current_val = current_features.get(name)
                # sqlite3.Row doesn't support .get(), use dictionary access directly
                # NULL values in SQLite will be returned as None in Python
                row_val = row[name]
                
                # Skip None values
                if current_val is None or row_val is None:
                    continue
                
                current_vec.append(float(current_val))
                row_vec.append(float(row_val))
            
            if len(current_vec) == 0:
                return self._time_decay_score(row, plan)
            
            # Calculate cosine similarity
            current_vec = np.array(current_vec)
            row_vec = np.array(row_vec)
            
            # Normalize
            current_norm = np.linalg.norm(current_vec)
            row_norm = np.linalg.norm(row_vec)
            
            if current_norm == 0 or row_norm == 0:
                return self._time_decay_score(row, plan)
            
            cosine_sim = np.dot(current_vec, row_vec) / (current_norm * row_norm)
            
            # Similarity score (0-1 range)
            similarity_score = (cosine_sim + 1) / 2  # Map [-1, 1] to [0, 1]
            
            # Combine with time decay
            time_score = self._time_decay_score(row, plan)
            
            # Weighted average: similarity 70%, time 30%
            final_score = 0.7 * similarity_score + 0.3 * time_score
            
            return final_score
            
        except Exception as e:
            logger.debug(f"Failed to calculate similarity: {e}, using time decay")
            return self._time_decay_score(row, plan)
    
    def _time_decay_score(self, row: sqlite3.Row, plan: RetrievalPlan) -> float:
        """Time decay score"""
        try:
            from datetime import datetime
            import math
            
            # Features table uses datetime column (format: YYYY-MM-DD)
            datetime_str = row['datetime']
            row_time = datetime.fromisoformat(datetime_str)
            plan_time = datetime.fromisoformat(plan.time_end.replace('Z', '+00:00'))
            days_diff = abs((plan_time - row_time).days)
            
            # Time decay: exp(-days/14), higher weight within 14 days
            time_score = math.exp(-days_diff / 14.0)
            
            return time_score
        except Exception:
            return 0.5  # Default score

