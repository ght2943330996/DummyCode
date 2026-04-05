# FAISS 持久化向量存储

## 概述

本项目现在使用 **FAISS (Facebook AI Similarity Search)** 作为向量存储引擎，替代了原来的内存存储。

### 主要改进

1. **持久化存储** - 向量计算一次后保存在本地，下次启动直接加载
2. **增量更新** - 只对新文件或修改过的文件重新向量化
3. **高效检索** - FAISS 提供毫秒级的向量相似度搜索
4. **文件变更检测** - 自动检测文件的修改、新增、删除

---

## 需要安装的依赖

### 方法1：使用 Conda（推荐）

```bash
conda install -c pytorch faiss-cpu
```

如果需要 GPU 加速：

```bash
conda install -c pytorch faiss-gpu
```

### 方法2：使用 pip

```bash
pip install faiss-cpu
```

**注意**：pip 安装的 faiss 在某些平台可能不稳定，建议使用 conda。

---

## 支持的文件格式

- **文本**: `.md`, `.txt`
- **PDF**: `.pdf`
- **Excel**: `.xlsx`, `.xlsm`, `.xltx`, `.xltm`, `.xls`
- **CSV**: `.csv`

## 存储结构

```
knowledge/
├── OpenAI_o1.md              # Markdown 文档
├── API_Reference.pdf         # PDF 文档
├── data.xlsx                 # Excel 表格
├── records.csv               # CSV 数据
└── embedding/                # 向量化缓存目录
    ├── index.json            # 文件索引信息（修改时间、hash）
    ├── vectors.faiss         # FAISS 索引文件（二进制）
    └── metadata.json         # 向量元数据（文本内容）
```

---

## 工作流程

### 首次运行

1. 扫描 `knowledge/` 目录中的支持文件
2. 对每个文件进行切片
3. 调用 Embedding API 生成向量
4. 保存到 `knowledge/embedding/`
5. **耗时**：取决于文件数量和 API 速度

### 后续运行

1. 加载 `knowledge/embedding/index.json`
2. 检查每个文件是否被修改（对比 mtime + hash）
3. 如果文件未变更，直接从 `vectors.faiss` 加载
4. 如果有新文件或修改，只对这些文件重新向量化
5. **耗时**：毫秒级（无需 API 调用）

---

## 触发重新向量化的条件

以下情况会自动重新向量化：

| 情况                 | 行为                 |
| -------------------- | -------------------- |
| 新增支持格式的文件   | 只对新文件向量化     |
| 修改已有文件         | 只对该文件重新向量化 |
| 删除已有文件         | 从索引中移除         |
| 更换 embedding 模型  | 重新向量化所有文件   |
| 删除 embedding/ 目录 | 完全重建索引         |

---

## 手动管理索引

### 完全重建索引

如果需要重建所有索引，删除 embedding 目录即可：

```bash
# 在项目根目录执行
rm -rf knowledge/embedding
```

下次运行时会自动重建。

### 查看索引统计

启动时控制台会输出：

```
==================================================
[FAISS Retriever] 初始化...
模型: BAAI/bge-m3
知识库: E:\work\demo\agent\llm-mcp-rag-python\knowledge
缓存目录: E:\work\demo\agent\llm-mcp-rag-python\knowledge\embedding
==================================================
[FAISS] 索引已是最新，从本地加载...
==================================================
[FAISS] 索引统计:
  总向量数: 128
  向量维度: 1024
  来源文件: 10 个
==================================================
```

---

## 代码架构

### 核心类

| 文件                      | 类                   | 职责                            |
| ------------------------- | -------------------- | ------------------------------- |
| `vector_store_faiss.py` | `FAISSVectorStore` | FAISS 索引管理、保存/加载、搜索 |
| `index_manager.py`      | `IndexManager`     | 文件变更检测、索引元数据管理    |
| `embedding_faiss.py`    | `FAISSRetriever`   | 高层接口，协调向量化流程        |

使用示例

```python
from pathlib import Path
from embedding_faiss import FAISSRetriever

# 初始化检索器
retriever = FAISSRetriever(
    embedding_model="BAAI/bge-m3",
    knowledge_dir=Path("knowledge")
)

# 加载缓存或执行向量化
await retriever.initialize()

# 检索相关内容
results = await retriever.retrieve("Claude 模型的特点", top_k=3)
for r in results:
    print(f"相关度: {r['score']:.4f}")
    print(f"内容: {r['document'][:100]}...")
    print(f"来源: {r['source_file']}")
```

---

## 性能对比

| 指标     | 旧方案（实时向量化）     | 新方案（FAISS 缓存）    |
| -------- | ------------------------ | ----------------------- |
| 启动时间 | 10-30 秒（取决于文件数） | < 1 秒                  |
| API 调用 | 每次启动都调用           | 仅首次或文件变更时      |
| 内存占用 | 所有向量在内存           | FAISS 优化存储          |
| 搜索速度 | 线性扫描 O(n)            | FAISS 近似搜索 O(log n) |

---

## 注意事项

1. **不要手动修改 embedding/ 目录下的文件**，可能导致索引损坏
2. **更换 embedding 模型**时会自动重建索引
3. **文件过大**（>10MB）可能导致内存不足，建议分割大文件
4. **FAISS 版本**建议使用 conda 安装的稳定版本
