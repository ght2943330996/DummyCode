import json
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class FileInfo:
    """文件信息"""
    path: str           # 相对路径
    mtime: float        # 修改时间
    size: int           # 文件大小
    hash: str           # 内容hash
    chunk_count: int = 0    # 切片数量


class IndexManager:
    """
    管理知识库文件的索引信息
    用于检测文件变更，决定哪些文件需要重新向量化
    """

    def __init__(
        self,
        knowledge_dir: Path,
        embedding_dir: Path,
        supported_extensions: set = None
    ):
        self.knowledge_dir = Path(knowledge_dir)
        self.embedding_dir = Path(embedding_dir)
        self.index_file = self.embedding_dir / "index.json"
        self.index_data = self._load_index()

        # 支持的文件扩展名
        self.supported_extensions = supported_extensions or {'.md', '.txt', '.pdf', '.xlsx', '.xlsm', '.xltx', '.xltm', '.xls', '.csv'}

    def _load_index(self) -> dict:
        """加载索引文件"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[警告] 加载索引文件失败: {e}，将创建新索引")
        return {
            "version": "1.0",
            "embedding_model": "",
            "files": {}
        }

    def save_index(self):
        """保存索引到文件"""
        self.embedding_dir.mkdir(parents=True, exist_ok=True)
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.index_data, f, ensure_ascii=False, indent=2)

    def _calc_file_hash(self, file_path: Path) -> str:
        """计算文件内容 hash"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]  # 取前16位足够用了

    def _get_file_info(self, file_path: Path) -> Optional[FileInfo]:
        """获取文件信息"""
        if not file_path.exists():
            return None

        stat = file_path.stat()
        return FileInfo(
            path=str(file_path.relative_to(self.knowledge_dir)),
            mtime=stat.st_mtime,
            size=stat.st_size,
            hash=self._calc_file_hash(file_path)
        )

    def scan_files(self) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        扫描知识库，检测文件变更

        Returns:
            (新增文件列表, 修改文件列表, 删除文件列表)
        """
        # 获取当前所有支持的文件
        current_files: set = set()
        for ext in self.supported_extensions:
            current_files.update(self.knowledge_dir.rglob(f"*{ext}"))

        # 获取索引中的文件
        indexed_files = {
            self.knowledge_dir / path
            for path in self.index_data.get("files", {}).keys()
        }

        new_files = []
        modified_files = []
        deleted_files = []

        # 检查新增和修改
        for file_path in current_files:
            rel_path = str(file_path.relative_to(self.knowledge_dir))
            file_info = self._get_file_info(file_path)

            if not file_info:
                continue

            if rel_path not in self.index_data.get("files", {}):
                # 新增文件
                new_files.append(file_path)
            else:
                # 检查是否修改
                old_info = self.index_data["files"][rel_path]
                if (file_info.mtime != old_info.get("mtime") or
                    file_info.hash != old_info.get("hash")):
                    modified_files.append(file_path)

        # 检查删除
        for file_path in indexed_files:
            if not file_path.exists():
                deleted_files.append(file_path)

        return new_files, modified_files, deleted_files

    def update_file(self, file_path: Path, chunk_count: int = 0):
        """更新文件索引信息"""
        rel_path = str(file_path.relative_to(self.knowledge_dir))
        file_info = self._get_file_info(file_path)
        if file_info:
            file_info.chunk_count = chunk_count
            self.index_data["files"][rel_path] = asdict(file_info)

    def remove_file(self, file_path: Path):
        """从索引中移除文件"""
        rel_path = str(file_path.relative_to(self.knowledge_dir))
        if rel_path in self.index_data.get("files", {}):
            del self.index_data["files"][rel_path]

    def get_embedding_model(self) -> str:
        """获取索引使用的 embedding 模型"""
        return self.index_data.get("embedding_model", "")

    def set_embedding_model(self, model: str):
        """设置 embedding 模型"""
        self.index_data["embedding_model"] = model

    def get_all_indexed_files(self) -> List[str]:
        """获取所有已索引的文件路径"""
        return list(self.index_data.get("files", {}).keys())

    def is_fresh(self, model: str) -> bool:
        """
        检查索引是否最新（没有变更，且模型一致）

        Returns:
            True: 索引最新，可以直接加载缓存
            False: 有变更，需要重新向量化
        """
        # 检查模型是否一致
        if self.get_embedding_model() != model:
            print(f"[索引] Embedding 模型变更: {self.get_embedding_model()} -> {model}")
            return False

        # 检查文件变更
        new_files, modified_files, deleted_files = self.scan_files()

        if new_files:
            print(f"[索引] 发现 {len(new_files)} 个新增文件")
        if modified_files:
            print(f"[索引] 发现 {len(modified_files)} 个修改文件")
        if deleted_files:
            print(f"[索引] 发现 {len(deleted_files)} 个删除文件")

        return len(new_files) == 0 and len(modified_files) == 0 and len(deleted_files) == 0
