from typing import List, Dict, Any, Optional
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
from util import log_title

# 自动加载.env
load_dotenv()


#api文档中stream中的ToolCall类
class ToolCall:
    #OpenAI ToolCall类
    def __init__(self, id: str, function: Dict[str, str]):
        self.id = id
        self.function = function


class ChatOpenAI:
    #定义ChatOpenAI类"
    def __init__(self, model: str, system_prompt: str = '', tools: List[Dict] = None, context: str = ''):  #context留给RAG注入
        # 初始化OpenAI客户端
        self.llm = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_BASE_URL'),
        )
        self.model = model
        self.messages: List[Dict[str, Any]] = []  # 对话历史
        self.tools = tools or []  # 可用的工具列表（MCP格式）

        # 可选：添加系统提示词
        if system_prompt:
            self.messages.append({'role': 'system', 'content': system_prompt})
        # 可选：添加上下文信息
        if context:
            self.messages.append({'role': 'user', 'content': context})

    async def chat(self, prompt: Optional[str] = None) -> Dict[str, Any]:  #prompt可选,返回字典
        #定义chat方法，处理用户输入并返回模型响应
        log_title('CHAT')

        # 步骤1: 添加用户消息到对话框
        if prompt:
            self.messages.append({'role': 'user', 'content': prompt})

        # 步骤2: 发送流式API请求,逐块chunk返回结果（如果不采用流式传输则返回response全部结果）
        stream = self.llm.chat.completions.create(              #调用OpenAI的chat.completions接口
            model=self.model,
            messages=self.messages,
            stream=True,
            tools=self._get_tools_definition() if self.tools else None,
        )

        content = ""
        tool_calls: List[ToolCall] = []
        log_title('RESPONSE')

        # 步骤3: 逐块处理流式接收响应
        for chunk in stream:
            delta = chunk.choices[0].delta     #delta会带来两个参数，需要分别处理

            # 处理普通Content文本内容
            if delta.content:
                content_chunk = delta.content or ""
                content += content_chunk
                sys.stdout.write(content_chunk)  #连续打印不换行
                sys.stdout.flush()      #刷新缓冲区,确保立即显示

            # 处理ToolCall工具调用
            if delta.tool_calls:
                for tool_call_chunk in delta.tool_calls:
                    # 初始化新的工具调用，append新的ToolCall对象
                    if len(tool_calls) <= tool_call_chunk.index:
                        tool_calls.append(ToolCall(id='', function={'name': '', 'arguments': ''}))

                    current_call = tool_calls[tool_call_chunk.index]
                    if tool_call_chunk.id:
                        current_call.id += tool_call_chunk.id
                    if tool_call_chunk.function and tool_call_chunk.function.name:
                        current_call.function['name'] += tool_call_chunk.function.name
                    if tool_call_chunk.function and tool_call_chunk.function.arguments:
                        current_call.function['arguments'] += tool_call_chunk.function.arguments

        # 步骤4: messages是用于给LLM记忆的，因为每次调用llm他都会忘记历史对话，所以必须把所有历史对话都放到messages中
        self.messages.append({
            'role': 'assistant',
            'content': content,
            'tool_calls': [
                {
                    'id': call.id,
                    'type': 'function',
                    'function': call.function
                }
                for call in tool_calls
            ] if tool_calls else None
        })
        """
        返回content和tool_calls，以便agent调用
        """
        #返回上下文和工具调用，以便agent调用
        return {
            'content': content,
            'toolCalls': tool_calls,
        }

    def append_tool_result(self, tool_call_id: str, tool_output: str):
        #添加工具执行结果到对话框
        self.messages.append({
            'role': 'tool',
            'content': tool_output,
            'tool_call_id': tool_call_id
        })

    def _get_tools_definition(self) -> List[Dict[str, Any]]:
        #转换MCP工具为OpenAI工具定义
        return [
            {
                'type': 'function',
                'function': {
                    'name': tool['name'],
                    'description': tool['description'],
                    'parameters': tool['inputSchema'],
                }
            }
            for tool in self.tools
        ]
