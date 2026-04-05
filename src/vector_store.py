from typing import List, Dict
import math


#向量储存项
class VectorStoreItem:
    def __init__(self, embedding: List[float], document: str):
        self.embedding = embedding                #向量
        self.document = document                  #文档内容


#向量储存类
class VectorStore:
    def __init__(self):
        self.vector_store: List[VectorStoreItem] = []

    #添加向量项到存储
    async def add_item(self, item: VectorStoreItem):
        self.vector_store.append(item)

    #搜索最相似的向量
    async def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict]:
        scores = [
            {
                'score': self._cosine_similarity(item.embedding, query_embedding),
                'document': item.document
            }
            for item in self.vector_store
        ]
        return sorted(scores, key=lambda x: x['score'], reverse=True)[:top_k]

    #计算余弦相似度
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm_a = math.sqrt(sum(a * a for a in v1))
        norm_b = math.sqrt(sum(b * b for b in v2))
        return dot_product / (norm_a * norm_b)

