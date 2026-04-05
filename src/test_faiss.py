"""
FAISS 向量存储测试脚本
用于验证 FAISS 实现是否正常工作
"""

import asyncio
import sys
from pathlib import Path

# 确保可以导入 src 目录下的模块
sys.path.insert(0, str(Path(__file__).parent))

from embedding_faiss import FAISSRetriever


async def test_faiss():
    """测试 FAISS 向量存储"""
    print("=" * 60)
    print("FAISS 向量存储测试")
    print("=" * 60)

    # 测试 1: 检查 faiss 是否安装
    print("\n[测试 1] 检查 FAISS 安装...")
    try:
        import faiss
        print(f"✓ FAISS 已安装，版本: {faiss.__version__}")
    except ImportError:
        print("✗ FAISS 未安装！")
        print("  请运行: conda install -c pytorch faiss-cpu")
        return False

    # 测试 2: 初始化检索器
    print("\n[测试 2] 初始化 FAISS 检索器...")
    knowledge_dir = Path(__file__).parent.parent / "knowledge"

    if not knowledge_dir.exists():
        print(f"✗ 知识库目录不存在: {knowledge_dir}")
        print("  请确保 knowledge/ 目录存在且有文件")
        return False

    retriever = FAISSRetriever(
        embedding_model="BAAI/bge-m3",
        knowledge_dir=knowledge_dir
    )
    print("✓ 检索器初始化成功")

    # 测试 3: 加载/构建索引
    print("\n[测试 3] 加载/构建索引...")
    try:
        is_fresh = await retriever.initialize()
        if is_fresh:
            print("✓ 从缓存加载成功")
        else:
            print("✓ 索引构建完成")
    except Exception as e:
        print(f"✗ 索引构建失败: {e}")
        return False

    # 测试 4: 检查索引统计
    print("\n[测试 4] 索引统计...")
    stats = retriever.vector_store.get_stats()
    print(f"  总向量数: {stats['total_vectors']}")
    print(f"  向量维度: {stats['dimension']}")
    print(f"  来源文件: {len(stats['source_files'])} 个")

    if stats['total_vectors'] == 0:
        print("\n  ⚠ 知识库为空或文件读取失败")
        return False

    # 测试 5: 执行检索
    print("\n[测试 5] 执行检索测试...")
    test_queries = [
        "Claude 模型",
        "GPT-4",
        "DeepSeek",
    ]

    for query in test_queries:
        print(f"\n  查询: '{query}'")
        try:
            results = await retriever.retrieve(query, top_k=2)
            if results:
                for i, r in enumerate(results, 1):
                    source = r.get('source_file', '未知')
                    preview = r['document'][:50].replace('\n', ' ')
                    print(f"    [{i}] 相关度: {r['score']:.4f} | 来源: {source}")
                    print(f"        {preview}...")
            else:
                print("    未找到相关结果")
        except Exception as e:
            print(f"    ✗ 检索失败: {e}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = asyncio.run(test_faiss())
    sys.exit(0 if success else 1)
