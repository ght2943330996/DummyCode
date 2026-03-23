from ast import Pass
import asyncio
import os
from pathlib import Path
from mcp_client import MCPClient
from agent import Agent
from embedding import EmbeddingRetriever
from util import log_title
from session import SessionManager


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


# 多会话命令行模式
async def session_chat():
    manager = SessionManager(
        model='deepseek-chat',
        system_prompt=(
            '你叫Dummy，是一个智能助手，拥有持久记忆能力。你必须严格遵守以下记忆规则：\n'
            '1. 【每次对话开始】调用 search_nodes 或 read_graph 查询与用户问题相关的已有记忆，优先基于记忆回答。\n'
            '2. 【每次对话结束前】将本轮对话中出现的重要信息（用户偏好、事实、计划、人名、结论等）'
            '用 create_entities 和 add_observations 存入知识图谱，用 create_relations 建立实体间关系。\n'
            '3. 若用户明确说"不用记"或信息无价值，则跳过存储。\n'
            '4. 存储时使用中文，实体名称简洁清晰。'
        ),
        default_extra_mcp_clients=[fetch_mcp, file_mcp],      #默认工具
    )

    print()
    print('=' * 80)
    print('  欢迎使用DummyCode！输入 /help 查看命令列表')
    print('=' * 80)

    # 启动时自动创建第一个会话
    # print('\n正在创建会话...')
    # await manager.create_session(name='默认会话')      #返回一个session
    # manager.print_help()   #通过实例调用方法时，Python 会自动将实例本身manager作为第一个参数传给方法

    # 启动时尝试从 index.json 恢复历史会话，否则创建默认会话
    restored = manager.load_index()
    if restored:
        sessions = manager.list_sessions()
        # print(f'\n历史会话：')
        manager.print_sessions(sessions)

        # 恢复后的当前会话处于懒初始化状态，在此立即连接 MCP
        current = manager.current_session
        if current and not current._initialized:
            print(f'正在连接会话「{current.name}」...')
            await current.init()
    else:
        print('\n无历史会话，正在创建默认会话...')
        await manager.create_session(name='默认会话')

    manager.print_help()

    try:  #捕获异常，ctrl+c退出循环
        while True:
            # 当前会话名称提示
            current = manager.current_session
            session_label = f'[{current.name}]' if current else '[无会话]'
            user_input = input(f'\n{session_label} 用户: ').strip()

            if not user_input:
                continue

            # 退出
            if user_input.lower() in ['exit', 'quit', '退出']:
                print('再见！')
                break

            # /help
            elif user_input.lower() == '/help':
                manager.print_help()

            # /new [名称]
            elif user_input.lower().startswith('/new'):
                parts = user_input.split(maxsplit=1)
                name = parts[1].strip() if len(parts) > 1 else None
                print(f'正在创建新会话{"：" + name if name else ""}...')
                try:
                    new_sess = await manager.create_session(name=name)
                    print(f'已创建并切换到会话: {new_sess.name}')
                except Exception as e:
                    print(f'创建会话失败: {e}')

            # /list
            elif user_input.lower() == '/list':
                log_title('所有会话')
                manager.print_sessions(manager.list_sessions())

            # /switch <编号>
            elif user_input.lower().startswith('/switch'):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print('用法: /switch <编号>  （编号来自 /list）')
                else:
                    try:
                        idx = int(parts[1].strip()) - 1
                        sessions = manager.list_sessions()
                        if idx < 0 or idx >= len(sessions):
                            print(f'编号超出范围，当前共 {len(sessions)} 个会话')
                        else:
                            target_id = sessions[idx]['session_id']
                            sess = await manager.switch_session(target_id)
                            print(f'已切换到会话: {sess.name}')
                    except ValueError:
                        print('请输入有效的数字编号')
                    except Exception as e:
                        print(f'切换失败: {e}')

            # /delete <编号>
            elif user_input.lower().startswith('/delete'):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print('用法: /delete <编号>  （编号来自 /list）')
                else:
                    try:
                        idx = int(parts[1].strip()) - 1
                        sessions = manager.list_sessions()
                        if idx < 0 or idx >= len(sessions):
                            print(f'编号超出范围，当前共 {len(sessions)} 个会话')
                        else:
                            target_id = sessions[idx]['session_id']
                            target_name = sessions[idx]['name']
                            confirm = input(f'确认删除会话「{target_name}」？(y/N): ').strip().lower()
                            if confirm == 'y':
                                await manager.delete_session(target_id)
                                print(f'已删除会话: {target_name}')
                                if not manager.current_session:
                                    print('所有会话已删除，正在创建新的默认会话...')
                                    await manager.create_session(name='默认会话')
                            else:
                                print('已取消')
                    except ValueError:
                        print('请输入有效的数字编号')
                    except Exception as e:
                        print(f'删除失败: {e}')

            # /clear
            elif user_input.lower() == '/clear':
                if manager.current_session:
                    manager.current_session.clear_conversation()
                    print('当前会话的对话历史已清空')
                else:
                    print('当前没有活跃会话')

            # /history
            elif user_input.lower() == '/history':
                if not manager.current_session:
                    print('当前没有活跃会话')
                else:
                    history = manager.current_session.get_conversation_history()
                    log_title('对话历史')
                    for msg in history:
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                        if content:
                            print(f"[{role}]: {content[:100]}..." if len(content) > 100 else f"[{role}]: {content}")

            # 普通对话
            else:
                if not manager.current_session:
                    print('当前没有活跃会话，请先用 /new 创建一个会话')
                    continue
                try:
                    await manager.invoke(user_input)
                    print()  # 流式输出后换行
                except Exception as e:
                    print(f'\n错误: {e}\n')

    except KeyboardInterrupt:
        print('\n\n对话被中断')
    finally:
        await manager.close_all()
        # print('所有连接已关闭')


async def main():
     # await continuous_chat()
    await session_chat()


if __name__ == '__main__':
    asyncio.run(main())

