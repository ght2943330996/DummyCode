from typing import List, Optional
import json
from mcp_client import MCPClient
from chat_openai import ChatOpenAI
from util import log_title


class Agent:
    #Agent代理类
    def __init__(self, model: str, mcp_clients: List[MCPClient], system_prompt: str = '', context: str = ''):
        self.mcp_clients = mcp_clients      # 定义MCP客户端列表
        self.llm: Optional[ChatOpenAI] = None
        self.model = model
        self.system_prompt = system_prompt    #系统提示词
        self.context = context                  #上下文注入


    #初始化Agent
    async def init(self):
        log_title('MCP列表')
        #循环为每个工作连接MCP服务器
        for client in self.mcp_clients:
            await client.init()

        #获取可用工具列表
        tools = []
        for client in self.mcp_clients:
            tools.extend(client.get_tools())

        self.llm = ChatOpenAI(self.model, self.system_prompt, tools, self.context)

    #关闭Agent,采用异步挂起,提升多个MCP客户端的关闭效率
    async def close(self):
        for client in self.mcp_clients:
            await client.close()

    #调用Agent处理请求
    async def invoke(self, prompt: str) -> str:
        if not self.llm:
            raise Exception('Agent 未初始化')

        response = await self.llm.chat(prompt)      #chat返回两个参数，content和toolCalls

        while True:
            if len(response['toolCalls']) > 0:    #是否需要使用工具
                for tool_call in response['toolCalls']:
                    # 查找对应的MCP客户端
                    mcp = None
                    for client in self.mcp_clients:
                        if any(t['name'] == tool_call.function['name'] for t in client.get_tools()):
                            mcp = client
                            break

                    if mcp:
                        print("\n")
                        log_title('MCP调用')
                        print(f"工具名: {tool_call.function['name']}")
                        print(f"参数: {tool_call.function['arguments']}")

                        result = await mcp.call_tool(
                            tool_call.function['name'],
                            json.loads(tool_call.function['arguments'])
                        )

                        # 将MCP结果转换为可序列化的格式,确保 LLM 能正确接收
                        if hasattr(result, 'content'):
                            # 处理MCP CallToolResult对象
                            result_content = []
                            for item in result.content:
                                if hasattr(item, 'text'):
                                    result_content.append(item.text)
                                else:
                                    result_content.append(str(item))
                            result_str = '\n'.join(result_content)
                        else:
                            result_str = str(result)

                        print(f"Result: {result_str}")
                        #把工具结果加入对话历史
                        self.llm.append_tool_result(tool_call.id, result_str)
                    else:
                        self.llm.append_tool_result(tool_call.id, 'Tool not found')

                # 工具调用后,继续对话，不传prompt，让LLM基于历史继续对话
                response = await self.llm.chat()
                continue

            # # 没有工具调用,结束对话
            # await self.close()
            # return response['content']


            # 连续对话(不关闭连接，保持会话状态)
            return response['content']


    #获取对话历史
    def get_conversation_history(self) -> List[dict]:
        if self.llm:
            return self.llm.messages
        return []
    #清空对话历史（保留系统提示词）
    def clear_conversation(self):
        if self.llm:
            system_messages = [msg for msg in self.llm.messages if msg.get('role') == 'system']
            self.llm.messages = system_messages


