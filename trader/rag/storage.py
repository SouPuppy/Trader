"""
RAG 存储层
数据库schema：rag_docs表 + rag_index管理表
"""
import sqlite3
import json
import uuid
from typing import List, Optional, Dict, Any
from pathlib import Path

from trader.config import DB_PATH
from trader.logger import get_logger
from trader.RAG.document import Document, DocumentSource

logger = get_logger(__name__)


class RAGStorage:
    """
    RAG文档存储
    
    数据库schema：
    - rag_docs: 存储文档
    - rag_index: 索引管理（用于跟踪embedding状态等）
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        初始化存储
        
        Args:
            db_path: 数据库路径（默认使用DB_PATH）
        """
        self.db_path = db_path or DB_PATH
        self._ensure_tables()
    
    def _ensure_tables(self):
        """确保表存在，如果不存在则创建"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建 rag_docs 表
        create_rag_docs_sql = """
        CREATE TABLE IF NOT EXISTS rag_docs (
            doc_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            time TEXT NOT NULL,
            symbol TEXT,
            title TEXT,
            text TEXT NOT NULL,
            metadata TEXT,  -- JSON格式
            hash TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
        """
        cursor.execute(create_rag_docs_sql)
        
        # 创建索引（提高查询性能）
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rag_docs_time ON rag_docs(time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rag_docs_symbol ON rag_docs(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rag_docs_source ON rag_docs(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rag_docs_hash ON rag_docs(hash)")
        
        # 创建 rag_index 管理表（用于跟踪embedding状态等）
        create_rag_index_sql = """
        CREATE TABLE IF NOT EXISTS rag_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            index_type TEXT NOT NULL,  -- 'embedding', 'fulltext', etc.
            index_status TEXT NOT NULL,  -- 'pending', 'indexed', 'failed'
            index_data TEXT,  -- JSON格式，存储索引相关数据
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (doc_id) REFERENCES rag_docs(doc_id) ON DELETE CASCADE,
            UNIQUE(doc_id, index_type)
        )
        """
        cursor.execute(create_rag_index_sql)
        
        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rag_index_doc_id ON rag_index(doc_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rag_index_status ON rag_index(index_status)")
        
        conn.commit()
        conn.close()
        logger.debug("RAG表已确保存在")
    
    def insert_document(self, doc: Document, check_duplicate: bool = True) -> bool:
        """
        插入文档
        
        Args:
            doc: 文档对象
            check_duplicate: 是否检查重复（基于hash）
            
        Returns:
            bool: 是否成功插入（如果重复则返回False）
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # 检查重复
            if check_duplicate and doc.hash:
                cursor.execute("SELECT doc_id FROM rag_docs WHERE hash = ?", (doc.hash,))
                existing = cursor.fetchone()
                if existing:
                    logger.debug(f"文档已存在（hash={doc.hash}），跳过插入")
                    conn.close()
                    return False
            
            # 插入文档
            insert_sql = """
            INSERT OR REPLACE INTO rag_docs 
            (doc_id, source, time, symbol, title, text, metadata, hash, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """
            
            metadata_json = json.dumps(doc.metadata, ensure_ascii=False)
            
            cursor.execute(insert_sql, (
                doc.doc_id,
                doc.source.value,
                doc.time,
                doc.symbol,
                doc.title,
                doc.text,
                metadata_json,
                doc.hash,
            ))
            
            conn.commit()
            logger.debug(f"文档已插入: doc_id={doc.doc_id}, source={doc.source.value}")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"插入文档失败: {e}", exc_info=True)
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def insert_documents(self, docs: List[Document], check_duplicate: bool = True) -> int:
        """
        批量插入文档
        
        Args:
            docs: 文档列表
            check_duplicate: 是否检查重复
            
        Returns:
            int: 成功插入的数量
        """
        count = 0
        for doc in docs:
            if self.insert_document(doc, check_duplicate):
                count += 1
        logger.info(f"批量插入完成: {count}/{len(docs)}")
        return count
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """根据doc_id获取文档"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM rag_docs WHERE doc_id = ?", (doc_id,))
            row = cursor.fetchone()
            
            if row:
                metadata = json.loads(row['metadata']) if row['metadata'] else {}
                doc = Document(
                    doc_id=row['doc_id'],
                    source=DocumentSource(row['source']),
                    time=row['time'],
                    symbol=row['symbol'],
                    title=row['title'],
                    text=row['text'],
                    metadata=metadata,
                    hash=row['hash'],
                )
                return doc
            else:
                return None
        finally:
            conn.close()
    
    def get_documents(self, doc_ids: List[str]) -> List[Document]:
        """批量获取文档"""
        docs = []
        for doc_id in doc_ids:
            doc = self.get_document(doc_id)
            if doc:
                docs.append(doc)
        return docs
    
    def update_index_status(
        self,
        doc_id: str,
        index_type: str,
        status: str,
        index_data: Optional[Dict[str, Any]] = None,
    ):
        """
        更新索引状态
        
        Args:
            doc_id: 文档ID
            index_type: 索引类型（如'embedding'）
            status: 状态（'pending', 'indexed', 'failed'）
            index_data: 索引数据（可选）
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            index_data_json = json.dumps(index_data, ensure_ascii=False) if index_data else None
            
            cursor.execute("""
                INSERT OR REPLACE INTO rag_index 
                (doc_id, index_type, index_status, index_data, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'))
            """, (doc_id, index_type, status, index_data_json))
            
            conn.commit()
            logger.debug(f"索引状态已更新: doc_id={doc_id}, type={index_type}, status={status}")
        except sqlite3.Error as e:
            logger.error(f"更新索引状态失败: {e}", exc_info=True)
            conn.rollback()
        finally:
            conn.close()
    
    def get_index_status(self, doc_id: str, index_type: str) -> Optional[str]:
        """获取索引状态"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "SELECT index_status FROM rag_index WHERE doc_id = ? AND index_type = ?",
                (doc_id, index_type)
            )
            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            conn.close()
    
    def count_documents(self, source: Optional[DocumentSource] = None) -> int:
        """统计文档数量"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if source:
                cursor.execute("SELECT COUNT(*) FROM rag_docs WHERE source = ?", (source.value,))
            else:
                cursor.execute("SELECT COUNT(*) FROM rag_docs")
            return cursor.fetchone()[0]
        finally:
            conn.close()

