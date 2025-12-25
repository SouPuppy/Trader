"""
RAG 检索层
向量检索 + 硬过滤 + 排序
"""
import sqlite3
from typing import List, Tuple, Optional
from datetime import datetime

from trader.config import DB_PATH
from trader.rag import get_rag_logger
from trader.RAG.document import Document, DocumentSource
from trader.RAG.query import Query, QueryFilter
from trader.RAG.storage import RAGStorage

logger = get_rag_logger(__name__)


class RAGRetriever:
    """
    RAG检索器
    
    核心原则：
    - 向量检索负责"相关性"
    - 过滤负责"正确性"（时间过滤防止未来信息泄漏）
    - 排序：相似度 + 时间衰减（近的更重要）
    """
    
    def __init__(self, storage: Optional[RAGStorage] = None):
        """
        初始化检索器
        
        Args:
            storage: 存储对象（默认创建新实例）
        """
        self.storage = storage or RAGStorage()
    
    def retrieve(self, query: Query) -> List[Document]:
        """
        检索文档
        
        Args:
            query: 查询对象
            
        Returns:
            List[Document]: 检索到的文档列表（已排序）
        """
        # 1. 硬过滤（SQL WHERE）
        filtered_docs = self._filter_documents(query.filters)
        
        if not filtered_docs:
            logger.debug("过滤后没有文档")
            return []
        
        # 2. 向量检索（相似度计算）
        # TODO: 实现真正的向量检索（需要embedding模型）
        # 目前先用简单的文本匹配作为占位
        scored_docs = self._score_documents(filtered_docs, query.query_text)
        
        # 3. 排序（根据ranking_policy）
        sorted_docs = self._rank_documents(scored_docs, query.ranking_policy, query.filters)
        
        # 4. 返回top_k
        return sorted_docs[:query.top_k]
    
    def _filter_documents(self, filters: QueryFilter) -> List[Document]:
        """硬过滤文档（SQL WHERE）"""
        conn = sqlite3.connect(self.storage.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # 构建WHERE子句
            where_clause, params = filters.to_sql_where()
            
            sql = f"SELECT * FROM rag_docs {where_clause}"
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            docs = []
            for row in rows:
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
                docs.append(doc)
            
            logger.debug(f"过滤后得到 {len(docs)} 个文档")
            return docs
            
        finally:
            conn.close()
    
    def _score_documents(self, docs: List[Document], query_text: str) -> List[Tuple[Document, float]]:
        """
        计算文档相似度分数
        
        TODO: 实现真正的向量检索
        目前使用简单的文本匹配作为占位
        
        Args:
            docs: 文档列表
            query_text: 查询文本
            
        Returns:
            List[Tuple[Document, float]]: (文档, 相似度分数) 列表
        """
        scored = []
        query_lower = query_text.lower()
        
        for doc in docs:
            # 简单的文本匹配分数（占位实现）
            score = 0.0
            
            # 标题匹配
            if doc.title:
                title_lower = doc.title.lower()
                if query_lower in title_lower:
                    score += 0.5
                # 简单的关键词匹配
                query_words = set(query_lower.split())
                title_words = set(title_lower.split())
                common_words = query_words & title_words
                if common_words:
                    score += len(common_words) * 0.1
            
            # 正文匹配
            text_lower = doc.text.lower()
            if query_lower in text_lower:
                score += 0.3
            query_words = set(query_lower.split())
            text_words = set(text_lower.split())
            common_words = query_words & text_words
            if common_words:
                score += len(common_words) * 0.05
            
            # 符号匹配（如果查询中包含符号）
            if doc.symbol and doc.symbol.lower() in query_lower:
                score += 0.2
            
            scored.append((doc, score))
        
        return scored
    
    def _rank_documents(
        self,
        scored_docs: List[Tuple[Document, float]],
        ranking_policy: str,
        filters: QueryFilter,
    ) -> List[Document]:
        """
        排序文档
        
        Args:
            scored_docs: (文档, 相似度分数) 列表
            ranking_policy: 排序策略
            filters: 过滤条件（用于时间衰减计算）
            
        Returns:
            List[Document]: 排序后的文档列表
        """
        if ranking_policy == "similarity_only":
            # 只按相似度排序
            sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in sorted_docs]
        
        elif ranking_policy == "time_only":
            # 只按时间排序（最新的在前）
            sorted_docs = sorted(scored_docs, key=lambda x: x[0].time, reverse=True)
            return [doc for doc, _ in sorted_docs]
        
        elif ranking_policy == "similarity_time_decay":
            # 相似度 + 时间衰减
            # 时间越近，权重越高
            time_max = filters.time_max
            if not time_max:
                # 如果没有时间上限，只按相似度排序
                sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
                return [doc for doc, _ in sorted_docs]
            
            try:
                time_max_dt = datetime.strptime(time_max, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                # 如果时间格式不对，只按相似度排序
                sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
                return [doc for doc, _ in sorted_docs]
            
            # 计算综合分数：相似度 * 时间衰减因子
            final_scores = []
            for doc, similarity_score in scored_docs:
                try:
                    doc_time_dt = datetime.strptime(doc.time, "%Y-%m-%d %H:%M:%S")
                    # 计算时间差（天数）
                    time_diff = (time_max_dt - doc_time_dt).total_seconds() / 86400.0
                    # 时间衰减因子：1.0（当天）到0.5（7天前）
                    # 使用指数衰减：decay = exp(-time_diff / 7)
                    time_decay = math.exp(-time_diff / 7.0)
                    # 综合分数
                    final_score = similarity_score * (0.7 + 0.3 * time_decay)
                    final_scores.append((doc, final_score))
                except ValueError:
                    # 如果时间格式不对，只使用相似度
                    final_scores.append((doc, similarity_score))
            
            sorted_docs = sorted(final_scores, key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in sorted_docs]
        
        else:
            # 默认：只按相似度排序
            logger.warning(f"未知的排序策略: {ranking_policy}，使用similarity_only")
            sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in sorted_docs]

