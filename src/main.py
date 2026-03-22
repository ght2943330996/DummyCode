from ast import Pass
import asyncio
import os
from pathlib import Path
from mcp_client import MCPClient
from agent import Agent
from embedding import EmbeddingRetriever
from util import log_title


# 获取工作根目录
current_dir = Path(os.getcwd()).parent
memory_file_path = current_dir / 'memory' / 'memory_like.jsonl'

# 初始化MCP客户端
fetch_mcp = MCPClient('fetch', 'uvx', ['mcp-server-fetch'])
file_mcp = MCPClient('file', 'npx', ['-y', '@modelcontextprotocol/server-filesystem', str(current_dir/'test')])
memory_mcp = MCPClient('memory', 'npx', ['-y', '@modelcontextprotocol/server-memory'], {"MEMORY_FILE_PATH": str(memory_file_path)})





# 测试ChatOpenAI类
# async def main():
#     from chat_openai import ChatOpenAI
#     llm = ChatOpenAI('deepseek-reasoner', '你是一个智能新闻助手')    # 模型+系统提示词
#     response = await llm.chat('你好')
#     print(response['content'])
#     print(response['toolCalls'])


# 测试mcp服务，记忆功能
# async def main():
#     agent = Agent('deepseek-chat', [fetch_mcp, file_mcp, memory_mcp], '必须优先查询记忆回答问题,并更新记忆', '')
#     await agent.init()
#     tools = await agent.invoke("根据我的喜好，为我制定一个旅游计划，并保存在{current_dir}/test中")
#     print(tools)
#     await agent.close()


# 测试llm+mcp
# async def main():
#     agent = Agent('deepseek-chat', [fetch_mcp, file_mcp])
#     await agent.init()
#     response = await agent.invoke(
#         f"爬取https://www.datalearner.com/leaderboards的内容,"
#         f"在{current_dir}/knowledge中，每个模型创建一个md文件保存基本信息"
#     )
#     print(response)


# llm+mcp+rag
# async def main():
#     # prompt = f"根据knowledge文件的模型信息,对比claude_Opus_4.5和Gemini_3.0_Pro的优缺点,并给出两个模型的具体使用场景,把结果保存到{current_dir}/test中"
#     prompt = f"根据张三的信息,为他制定一个学习计划,把结果保存到{current_dir}/test中"

#     context = await retrieve_context(prompt)

#     # Agent    # 模型名称+系统提示词+上下文
#     agent = Agent('deepseek-chat', [fetch_mcp, file_mcp], '', context)
#     await agent.init()
#     response = await agent.invoke(prompt)
#     print(response)
#     await agent.close()


# RAG检索
async def retrieve_context(prompt: str) -> str:
    embedding_retriever = EmbeddingRetriever("BAAI/bge-m3")     #嵌入模型名称
    knowledge_dir = Path(current_dir) / 'knowledge'         #RAG知识库目录

    files = list(knowledge_dir.iterdir())
    for file in files:
        if file.is_file():
            content = file.read_text(encoding='utf-8')
            await embedding_retriever.embed_document(content)

    context_results = await embedding_retriever.retrieve(prompt)
    log_title('上下文')
    print(context_results)

    return '\n'.join(item['document'] for item in context_results)


#连续对话
async def continuous_chat():
    """连续对话模式 - 保持会话上下文，支持多轮对话"""
    agent = Agent(
        'deepseek-chat',
        [fetch_mcp, file_mcp, memory_mcp],
        '你是一个智能助手，可以查询和更新记忆。请优先查询用户的记忆信息来回答问题。',
        ''
    )

    await agent.init()

    print("\n" + "=" * 80)
    print("提示：输入 'exit' 或 'quit' 退出对话")
    print("提示：输入 'clear' 清空对话历史")
    print("提示：输入 'history' 查看对话历史")
    print("=" * 80 + "\n")

    try:
        while True:
            # 获取用户输入
            user_input = input("用户: ").strip()

            # 处理特殊命令
            if user_input.lower() in ['exit', 'quit', '退出']:
                print("再见！")
                break
            elif user_input.lower() == 'clear':
                agent.clear_conversation()
                print("对话历史已清空")
                continue
            elif user_input.lower() == 'history':
                history = agent.get_conversation_history()
                log_title('对话历史')
                for msg in history:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    if content:
                        print(f"[{role}]: {content[:100]}..." if len(content) > 100 else f"[{role}]: {content}")
                continue
            elif not user_input:
                continue

            try:
                response = await agent.invoke(user_input)
                print("\n")
            except Exception as e:
                print(f"\n错误: {e}\n")

    except KeyboardInterrupt:
        print("\n\n对话被中断")
    finally:
        await agent.close()
        print("连接已关闭")


async def main():
    await continuous_chat()



if __name__ == '__main__':
    asyncio.run(main())

