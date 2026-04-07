import os
import requests
import asyncio
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv

from chunk import DocumentChunker
from vector_store_faiss import FAISSVectorStore, VectorStoreItem
from index_manager import IndexManager
from read_doc import DocumentReader

# 自动加载.env
load_dotenv()


class FAISSRetriever:
    """基于 FAISS 的嵌入检索器"""

    def __init__(
        self,
        embedding_model: str,
        knowledge_dir: Path,
        chunker: DocumentChunker = None,
        dimension: int = 1024
    ):
        """
        Args:
            embedding_model: 嵌入模型名称，如 "BAAI/bge-m3"
            knowledge_dir: 知识库目录路径
            chunker: 文档切片器
            dimension: 向量维度（BAAI/bge-m3 是 1024）
        """
        self.embedding_model = embedding_model
        self.knowledge_dir = Path(knowledge_dir)
        self.embedding_dir = self.knowledge_dir / "embedding"
        self.chunker = chunker or DocumentChunker()

        # 支持的文件扩展名
        self.supported_extensions = {'.md', '.txt', '.pdf', '.xlsx', '.xlsm', '.xltx', '.xltm', '.xls', '.csv'}

        # 初始化组件
        self.vector_store = FAISSVectorStore(self.embedding_dir, dimension)
        self.index_manager = IndexManager(
            self.knowledge_dir,
            self.embedding_dir,
            supported_extensions=self.supported_extensions
        )
        self.doc_reader = DocumentReader()  # 文档读取器（支持多种格式）

        # 统计信息
        self.stats = {
            "processed_files": 0,
            "skipped_files": 0,
            "total_chunks": 0
        }

    async def initialize(self) -> bool:
        """
        初始化检索器

        Returns:
            True: 从缓存加载
            False: 需要重新向量化
        """
        print(f"\n{'='*50}")
        print("[FAISS Retriever] 初始化...")
        print(f"模型: {self.embedding_model}")
        print(f"知识库: {self.knowledge_dir}")
        print(f"缓存目录: {self.embedding_dir}")
        print('='*50)

        # 检查索引是否最新
        if self.index_manager.is_fresh(self.embedding_model):
            print("[FAISS] 索引已是最新，从本地加载...")
            self.vector_store.load()
            self._print_stats()
            return True

        # 需要更新索引
        print("[FAISS] 检测到变更，需要更新索引...")
        await self._update_index()
        return False

    async def _update_index(self):
        """更新索引（增量更新）"""
        # 加载已有索引（如果存在）
        has_existing = self.vector_store.load()

        if not has_existing:
            print("[FAISS] 创建新索引...")
            self.index_manager.index_data = {
                "version": "1.0",
                "embedding_model": self.embedding_model,
                "files": {}
            }

        # 扫描文件变更
        new_files, modified_files, deleted_files = self.index_manager.scan_files()

        # 处理删除的文件（从索引中移除）
        for file_path in deleted_files:
            print(f"[删除] {file_path.name}")
            self.index_manager.remove_file(file_path)

        # 处理修改的文件（重新向量化）
        for file_path in modified_files:
            print(f"[更新] {file_path.name}")
            # 从向量存储中移除旧数据
            rel_path = str(file_path.relative_to(self.knowledge_dir))
            self.vector_store.remove_by_source_file(rel_path)
            # 重新向量化
            await self._process_file(file_path)

        # 处理新增的文件
        for file_path in new_files:
            print(f"[新增] {file_path.name}")
            await self._process_file(file_path)

        # 如果没有文件变更，但模型变更，需要重新处理所有文件
        if not new_files and not modified_files and not deleted_files:
            if self.index_manager.get_embedding_model() != self.embedding_model:
                print(f"[FAISS] 模型变更，重新向量化所有文件...")
                # 清空旧索引
                self.vector_store.clear()
                # 处理所有文件
                # 获取所有支持的文件
                all_files: list = []
                for ext in self.supported_extensions:
                    all_files.extend(self.knowledge_dir.rglob(f"*{ext}"))
                for file_path in all_files:
                    print(f"[处理] {file_path.name}")
                    await self._process_file(file_path)

        # 保存索引
        self.vector_store.save()
        self.index_manager.set_embedding_model(self.embedding_model)
        self.index_manager.save_index()

        self._print_stats()

    async def _process_file(self, file_path: Path):
        """
        处理单个文件：读取、切片、向量化、存储

        Args:
            file_path: 文件路径
        """
        try:
            # 读取文件（支持多种格式）
            content = self.doc_reader.read_knowledge_file(file_path)
            if not content or not content.strip():
                print(f"  └─ [跳过] 文件为空或无法读取")
                return

            # 切片
            chunks = self.chunker.chunk(content)
            if not chunks:
                return

            # 向量化
            items = []
            rel_path = str(file_path.relative_to(self.knowledge_dir))

            for chunk in chunks:
                embedding = await self._embed(chunk)
                items.append(VectorStoreItem(
                    embedding=embedding,
                    document=chunk,
                    source_file=rel_path
                ))

            # 添加到向量存储
            self.vector_store.add_items(items)

            # 更新索引
            self.index_manager.update_file(file_path, len(chunks))

            # 更新统计
            self.stats["processed_files"] += 1
            self.stats["total_chunks"] += len(chunks)

            # print(f"  └─ 生成 {len(chunks)} 个切片")

        except Exception as e:
            print(f"  └─ [错误] {e}")

    async def _embed(self, text: str) -> List[float]:
        """
        调用 Embedding API

        Args:
            text: 文本内容

        Returns:
            向量
        """
        response = requests.post(
            f"{os.getenv('EMBEDDING_BASE_URL')}/embeddings",
            headers={
                'Content-Type': 'application/json',
                'Authorization': f"Bearer {os.getenv('EMBEDDING_KEY')}",
            },
            json={
                'model': self.embedding_model,
                'input': text,
            }
        )
        response.raise_for_status()
        data = response.json()
        return data['data'][0]['embedding']

    async def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        检索最相关的文档

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            结果列表
        """
        query_embedding = await self._embed(query)
        return self.vector_store.search(query_embedding, top_k)

    def _print_stats(self):
        """打印统计信息"""
        stats = self.vector_store.get_stats()
        print(f"\n{'='*50}")
        print("[FAISS] 索引统计:")
        print(f"  总向量数: {stats['total_vectors']}")
        print(f"  向量维度: {stats['dimension']}")
        print(f"  来源文件: {len(stats['source_files'])} 个")
        if self.stats["processed_files"] > 0:
            print(f"  本次处理: {self.stats['processed_files']} 个文件")
            print(f"  本次切片: {self.stats['total_chunks']} 个")
        print('='*50 + "\n")


# 使用继承的方法，兼容旧接口的包装类
class EmbeddingRetriever(FAISSRetriever):
    pass
