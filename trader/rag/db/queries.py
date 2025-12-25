"""
RAG Database Query Module
"""
import sqlite3
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from trader.config import DB_PATH
from trader.rag import get_rag_logger

logger = get_rag_logger(__name__)


def get_db_connection():
    """Get database connection"""
    return sqlite3.connect(DB_PATH)


def ensure_tables():
    """Ensure necessary tables exist"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Note: trends_features table is no longer used
    # Features are now retrieved directly from the features table
    
    # Create trade_history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_history (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          stock_code TEXT NOT NULL,
          trade_time TEXT NOT NULL,
          action TEXT NOT NULL,
          price REAL NOT NULL,
          volume REAL NOT NULL
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_stock_time ON trade_history(stock_code, trade_time)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_stock ON trade_history(stock_code)")
    
    # Check if raw_data table has necessary fields (for news retrieval)
    cursor.execute("PRAGMA table_info(raw_data)")
    columns = [col[1] for col in cursor.fetchall()]
    
    # Ensure analyzed field exists (for news analysis status)
    if 'analyzed' not in columns:
        logger.info("Adding analyzed field to raw_data table...")
        cursor.execute("ALTER TABLE raw_data ADD COLUMN analyzed INTEGER DEFAULT 0")
    
    conn.commit()
    conn.close()
    logger.info("RAG database tables ensured")

