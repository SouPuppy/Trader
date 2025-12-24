"""
TradeRetriever: Retrieve trade history candidates
"""
import sqlite3
from typing import List
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.rag.types import RetrievalPlan, Candidate, Document
from trader.rag.db.queries import get_db_connection
from trader.logger import get_logger

logger = get_logger(__name__)


class TradeRetriever:
    """Trade history retriever"""
    
    def retrieve(self, plan: RetrievalPlan) -> List[Candidate]:
        """
        Retrieve trade history candidates
        
        Args:
            plan: Retrieval plan
            
        Returns:
            List[Candidate]
        """
        need = plan.needs.get("trade_history")
        if not need or not need.enable:
            logger.debug("Trade history retrieval not enabled")
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
            
            # Time window filter
            conditions.append("trade_time BETWEEN ? AND ?")
            params.extend([plan.time_start, plan.time_end])
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            # Query SQL
            max_trades = plan.constraints.get("max_trades", need.recall_k)
            limit = min(need.recall_k, max_trades)
            
            query = f"""
                SELECT id, stock_code, trade_time, action, price, volume
                FROM trade_history
                WHERE {where_clause}
                ORDER BY trade_time DESC
                LIMIT ?
            """
            params.append(limit)
            
            logger.debug(f"Executing trade history retrieval query: {query[:200]}...")
            logger.debug(f"Query parameters: stock_code={plan.stock_code}, time_range=[{plan.time_start}, {plan.time_end}]")
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            logger.info(f"Trade history retrieval returned {len(rows)} records")
            
            # If no records found, check if there's any data for this stock_code or time range
            if len(rows) == 0:
                # Check if stock_code exists at all
                if plan.stock_code:
                    cursor.execute("SELECT COUNT(*) FROM trade_history WHERE stock_code = ?", (plan.stock_code,))
                    stock_count = cursor.fetchone()[0]
                    if stock_count == 0:
                        # Check what stock_codes exist
                        cursor.execute("SELECT DISTINCT stock_code FROM trade_history LIMIT 5")
                        existing_stocks = [r[0] for r in cursor.fetchall()]
                        logger.warning(f"No trade history found for stock_code={plan.stock_code}. "
                                     f"Available stock_codes in trade_history: {existing_stocks}")
                    else:
                        logger.warning(f"Found {stock_count} records for stock_code={plan.stock_code}, "
                                     f"but none in time range [{plan.time_start}, {plan.time_end}]")
                        # Check time range
                        cursor.execute("SELECT MIN(trade_time), MAX(trade_time) FROM trade_history WHERE stock_code = ?", 
                                     (plan.stock_code,))
                        time_range = cursor.fetchone()
                        if time_range[0]:
                            logger.info(f"Available time range for {plan.stock_code}: {time_range[0]} to {time_range[1]}")
                else:
                    # Check total count
                    cursor.execute("SELECT COUNT(*) FROM trade_history")
                    total_count = cursor.fetchone()[0]
                    logger.warning(f"No trade history found in time range [{plan.time_start}, {plan.time_end}]. "
                                 f"Total records in trade_history: {total_count}")
            
            # Convert to Document and Candidate
            for row in rows:
                try:
                    doc = self._row_to_document(row)
                    if doc:
                        score = self._calculate_recall_score(row, plan)
                        candidate = Candidate(
                            doc=doc,
                            recall_score=score,
                            recall_source="sql_filter"
                        )
                        candidates.append(candidate)
                except Exception as e:
                    logger.warning(f"Failed to convert trade record (id={row['id']}): {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Trade history retrieval failed: {e}", exc_info=True)
        finally:
            conn.close()
        
        logger.info(f"Trade history retrieval completed, returned {len(candidates)} candidates")
        return candidates
    
    def _row_to_document(self, row: sqlite3.Row) -> Document:
        """Convert database row to Document"""
        trade_id = row['id']
        stock_code = row['stock_code']
        trade_time = row['trade_time']
        action = row['action']
        price = row['price']
        volume = row['volume']
        
        payload = {
            "action": action,
            "price": price,
            "volume": volume
        }
        
        # Generate text description
        text = f"{action} {volume} @ {price}"
        
        doc = Document(
            doc_id=f"trade:{trade_id}",
            doc_type="trade_history",
            stock_code=stock_code,
            timestamp=trade_time,
            payload=payload,
            text=text
        )
        
        return doc
    
    def _calculate_recall_score(self, row: sqlite3.Row, plan: RetrievalPlan) -> float:
        """Calculate recall score"""
        # Simple rule: based on time decay
        try:
            from datetime import datetime
            row_time = datetime.fromisoformat(row['trade_time'].replace('Z', '+00:00'))
            plan_time = datetime.fromisoformat(plan.time_end.replace('Z', '+00:00'))
            days_diff = abs((plan_time - row_time).days)
            
            # Time decay: exp(-days/30), higher weight within 30 days
            import math
            time_score = math.exp(-days_diff / 30.0)
            
            return time_score
        except Exception:
            return 0.5  # Default score

