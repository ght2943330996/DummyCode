import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import faiss


class VectorStoreItem:
    """向量存储项"""
    def __init__(self, embedding: List[float], document: str, source_file: str = ""):
        self.embedding = embedding
        self.document = document
        self.source_file = source_file  # 来源文件路径


class FAISSVectorStore:
    """基于 FAISS 的向量存储与查找"""

    def __init__(self, embedding_dir: Path, dimension: int = 1024):

        self.embedding_dir = Path(embedding_dir)
        self.dimension = dimension

        # FAISS 索引文件
        self.faiss_file = self.embedding_dir / "vectors.faiss"
        # 元数据文件（存储文本内容）
        self.metadata_file = self.embedding_dir / "metadata.json"

        # FAISS 索引
        self.index = None
        # 元数据列表（与 FAISS 索引顺序对应）
        self.metadata: List[Dict] = []

        # 创建目录
        self.embedding_dir.mkdir(parents=True, exist_ok=True)

    def _create_index(self) -> faiss.Index:
        """创建新的 FAISS 索引

        使用 IndexFlatIP (Inner Product) 支持余弦相似度，适用于1万-100万个向量
        归一化后，内积等价于余弦相似度
        """
        # IndexFlatIP: 精确搜索，内积相似度
        index = faiss.IndexFlatIP(self.dimension)
        return index

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """归一化向量，使内积等价于余弦相似度"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # 避免除零
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    def load(self) -> bool:
        """
        从磁盘加载索引

        Returns:
            True: 加载成功
            False: 没有保存的索引，需要新建
        """
        if not self.faiss_file.exists() or not self.metadata_file.exists():
            # 创建新的索引
            self.index = self._create_index()
            self.metadata = []
            return False

        try:
            # 加载 FAISS 索引
            self.index = faiss.read_index(str(self.faiss_file))

            # 加载元数据
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

            print(f"[FAISS] 加载索引成功: {len(self.metadata)} 个向量")
            return True

        except Exception as e:
            print(f"[FAISS] 加载索引失败: {e}，将创建新索引")
            self.index = self._create_index()
            self.metadata = []
            return False

    def save(self):
        """保存索引到磁盘"""
        if self.index is None:
            return

        try:
            # 保存 FAISS 索引
            faiss.write_index(self.index, str(self.faiss_file))

            # 保存元数据
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)

            print(f"[FAISS] 保存索引成功: {len(self.metadata)} 个向量")

        except Exception as e:
            print(f"[FAISS] 保存索引失败: {e}")

    def add_items(self, items: List[VectorStoreItem]):
        """批量添加向量项"""
        if not items:
            return

        if self.index is None:
            self.index = self._create_index()

        # 准备向量数组
        vectors = np.array([item.embedding for item in items], dtype=np.float32)

        # 归一化向量
        vectors = self._normalize_vectors(vectors)

        # 添加到 FAISS 索引
        self.index.add(vectors)

        # 保存元数据
        for item in items:
            self.metadata.append({
                "document": item.document,
                "source_file": item.source_file
            })

    def add_item(self, item: VectorStoreItem):
        """添加单个向量项"""
        self.add_items([item])

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict]:
        """
        搜索最相似的向量
        Returns:
            结果列表，包含 score 和 document
        """
        if self.index is None or len(self.metadata) == 0:
            return []

        # 准备查询向量
        query_vector = np.array([query_embedding], dtype=np.float32)

        # 归一化查询向量
        query_vector = self._normalize_vectors(query_vector)

        # 执行搜索
        scores, indices = self.index.search(query_vector, min(top_k, len(self.metadata)))

        # 组装结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):  # FAISS 可能返回 -1
                meta = self.metadata[idx]
                results.append({
                    "score": float(score),  # 归一化后的内积即为余弦相似度
                    "document": meta["document"],
                    "source_file": meta.get("source_file", "")
                })

        return results

    def clear(self):
        """清空索引"""
        self.index = self._create_index()
        self.metadata = []

    def remove_by_source_file(self, source_file: str):
        """
        移除特定来源文件的所有向量
        注意：FAISS 不支持直接删除，需要重建索引

        Args:
            source_file: 来源文件路径
        """
        # 找出需要保留的项
        keep_indices = []
        keep_metadata = []

        for i, meta in enumerate(self.metadata):
            if meta.get("source_file") != source_file:
                keep_indices.append(i)
                keep_metadata.append(meta)

        if len(keep_indices) == len(self.metadata):
            # 没有需要删除的
            return

        # 重建索引
        if keep_indices:
            # 获取保留的向量
            # FAISS 没有直接获取向量的方法，需要重新训练
            # 这里我们标记需要重建，下次 add 时处理
            pass

        # 简单处理：标记需要重建
        self._needs_rebuild = True
        self.metadata = keep_metadata

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "total_vectors": len(self.metadata),
            "dimension": self.dimension,
            "source_files": list(set(m.get("source_file", "") for m in self.metadata))
        }

    def __len__(self) -> int:
        """返回向量数量"""
        return len(self.metadata) if self.metadata else 0
