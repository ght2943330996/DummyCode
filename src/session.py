import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from agent import Agent
from mcp_client import MCPClient
from util import log_title


# 工作根目录（src 的上一级）
_ROOT_DIR = Path(os.getcwd()).parent
_MEMORY_DIR = _ROOT_DIR / 'memory' / 'sessions'
_INDEX_FILE = _ROOT_DIR / 'memory' / 'index.json'


#单个会话，封装独立的 Agent 和专属 Memory MCP
class Session:
    def __init__(
        self,
        session_id: str,
        name: str,
        model: str = '',           #前面传过来了model和system_prompt
        system_prompt: str = '',
        extra_mcp_clients: Optional[List[MCPClient]] = None,
        context: str = '',
        ):
        self.session_id = session_id
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.context = context
        self.created_at: str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')     #用于时间管理
        self.extra_mcp_clients: List[MCPClient] = extra_mcp_clients or []

        # 每个会话拥有独立的 memory 文件
        _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        self.memory_file: Path = _MEMORY_DIR / f'{self.name}.jsonl'

        # 专属 Memory MCP 客户端
        self.memory_mcp = MCPClient(
            name=f'memory_{session_id}',
            command='npx',
            args=['-y', '@modelcontextprotocol/server-memory'],
            env={'MEMORY_FILE_PATH': str(self.memory_file)},
        )

        # 组合所有 MCP 客户端
        all_mcp_clients = [self.memory_mcp] + self.extra_mcp_clients

        # 创建 Agent
        self.agent = Agent(
            model=self.model,
            mcp_clients=all_mcp_clients,
            system_prompt=self.system_prompt,
            context=self.context,
        )

        self._initialized: bool = False



    #初始化会话（连接 MCP 服务器）
    async def init(self):
        if not self._initialized:
            await self.agent.init()
            self._initialized = True
            # log_title(f'会话已初始化: {self.name}')

    #关闭会话（断开 MCP 连接）
    async def close(self):
        if self._initialized:
            await self.agent.close()
            self._initialized = False
            # log_title(f'会话已关闭: {self.name}')



    #对话
    async def invoke(self, prompt: str) -> str:
        if not self._initialized:
            await self.init()
        return await self.agent.invoke(prompt)

    #清空对话历史
    def clear_conversation(self):
        self.agent.clear_conversation()

    #获取对话历史
    def get_conversation_history(self) -> List[dict]:
        return self.agent.get_conversation_history()


    #将会话信息转换为字典（用于持久化）
    def to_dict(self) -> dict:
        return {
            'session_id': self.session_id,
            'name': self.name,
            'model': self.model,
            'system_prompt': self.system_prompt,
            'created_at': self.created_at,
            'memory_file': str(self.memory_file),
            'initialized': self._initialized,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        extra_mcp_clients: Optional[List[MCPClient]] = None,
        context: str = '',
    ) -> 'Session':
        # 从index.json中的元数据恢复创建会话实例（尚未连接 MCP）
        session = cls.__new__(cls)
        session.session_id = data['session_id']
        session.name = data['name']
        session.model = data.get('model', '')
        session.system_prompt = data.get('system_prompt', '')
        session.context = context
        session.created_at = data.get('created_at', '')
        session.extra_mcp_clients = extra_mcp_clients or []

        _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        session.memory_file = Path(data['memory_file'])

        session.memory_mcp = MCPClient(
            name=f'memory_{session.session_id}',
            command='npx',
            args=['-y', '@modelcontextprotocol/server-memory'],
            env={'MEMORY_FILE_PATH': str(session.memory_file)},
        )

        all_mcp_clients = [session.memory_mcp] + session.extra_mcp_clients
        session.agent = Agent(
            model=session.model,
            mcp_clients=all_mcp_clients,
            system_prompt=session.system_prompt,
            context=session.context,
        )
        session._initialized = False
        return session


#会话管理器，负责创建、切换、关闭会话
class SessionManager:
    def __init__(
        self,
        model: str = '',
        system_prompt: str = '',
        default_extra_mcp_clients: Optional[List[MCPClient]] = None,
        ):
        self.model = model
        self.system_prompt = system_prompt
        self.default_extra_mcp_clients: List[MCPClient] = default_extra_mcp_clients or []

        self._sessions: Dict[str, Session] = {}      #id对应会话，这里并未实例化session
        self._current_session_id: Optional[str] = None  #当前会话id
        self._id_counter: int = 0

    #创建会话   主要是改变session_id和name
    async def create_session(
        self,
        name: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        extra_mcp_clients: Optional[List[MCPClient]] = None,   #除 memory_mcp 之外的额外 MCP 客户端
        context: str = '',
        auto_switch: bool = True,                              #创建后自动切换到该会话
        ) -> Session:

        session_id = self._generate_id()
        display_name = name or f'会话 {self._id_counter}'     #有name则使用name，否则使用f

        session = Session(
            session_id=session_id,
            name=display_name,
            model=model or self.model,
            system_prompt=system_prompt or self.system_prompt,
            extra_mcp_clients=extra_mcp_clients if extra_mcp_clients is not None else list(self.default_extra_mcp_clients),
            context=context,
        )

        # 立即初始化（连接 MCP）
        await session.init()

        self._sessions[session_id] = session        #将会话添加到会话管理器中,将字典中键 session_id 所对应的值设置为 session

        if auto_switch:
            self._current_session_id = session_id

        log_title(f'新会话已创建: {display_name}')
        return session


    def _generate_id(self) -> str:
        self._id_counter += 1
        ts = datetime.now().strftime('%Y%m%d%H%M%S')   #获取当前时间戳
        return f'session_{ts}_{self._id_counter}'


    # 保存当前会话元素到 index.json
    def save_index(self):
        data = {
            'current_session_id': self._current_session_id,
            'sessions': [s.to_dict() for s in self._sessions.values()],
        }
        _INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
        _INDEX_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

    #加载index.json中的会话元素
    def load_index(self) -> bool:
        if not _INDEX_FILE.exists():
            return False
        try:
            data = json.loads(_INDEX_FILE.read_text(encoding='utf-8'))
        except Exception as e:
            print(f'[警告] 读取 index.json 失败，将忽略历史会话: {e}')
            return False

        sessions_data = data.get('sessions', [])
        if not sessions_data:
            return False

        for s_data in sessions_data:
            session = Session.from_dict(
                s_data,
                extra_mcp_clients=list(self.default_extra_mcp_clients),
            )
            self._sessions[session.session_id] = session

        # 恢复上次的当前会话，若已不存在则取最后一个
        saved_current = data.get('current_session_id')
        if saved_current and saved_current in self._sessions:
            self._current_session_id = saved_current
        elif self._sessions:
            self._current_session_id = list(self._sessions.keys())[-1]

        # 同步计数器，避免 ID 冲突
        self._id_counter = len(self._sessions)
        log_title('历史会话')
        return True


    #/help
    def print_help(self):
        print("""
        ╔══════════════════════════════════════════════════════════════════════════════╗
        ║                      欢迎来到DummyCode!  命令列表                            ║
        ╠══════════════════════════════════════════════════════════════════════════════╣
        ║  /new  [名称]      创建新会话（可选传名称，默认自动命名）                    ║
        ║  /list             列出所有会话                                              ║
        ║  /switch <编号>    按列表编号切换到指定会话                                  ║
        ║  /delete <编号>    按列表编号删除指定会话                                    ║
        ║  /clear            清空当前会话的对话历史                                    ║
        ║  /history          查看当前会话的对话历史                                    ║
        ║  /help             显示此帮助信息                                            ║
        ║  /quit             退出程序                                                  ║
        ╚══════════════════════════════════════════════════════════════════════════════╝
        """)


    #/list字典化
    def list_sessions(self) -> List[dict]:
        result = []
        for session in self._sessions.values():
            info = session.to_dict()
            info['is_current'] = (session.session_id == self._current_session_id)
            result.append(info)
        return result

    #/list
    def print_sessions(self, sessions: List[dict]):
        if not sessions:
            print('（暂无会话）')
            return
        print()
        for i, s in enumerate(sessions, 1):
            mark = ' ◀ 当前' if s['is_current'] else ''
            status = '已连接' if s['initialized'] else '未连接'
            print(f"  {i}. [{status}] {s['name']}  |  创建于 {s['created_at']}  |  ID: {s['session_id']}{mark}")
        print()

    #/switch <编号>
    async def switch_session(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            raise ValueError(f'会话不存在: {session_id}')

        session = self._sessions[session_id]

        # 如果目标会话尚未初始化，则初始化
        if not session._initialized:
            await session.init()

        self._current_session_id = session_id
        log_title(f'已切换到会话: {session.name}')
        return session

    #删除会话    默认保留memory文件
    async def delete_session(self, session_id: str, delete_memory_file: bool = False):
        if session_id not in self._sessions:
            raise ValueError(f'会话不存在: {session_id}')

        session = self._sessions.pop(session_id)

        # 关闭 MCP 连接
        await session.close()

        # 可选：删除 memory 文件
        if delete_memory_file and session.memory_file.exists():
            session.memory_file.unlink()
            print(f'已删除 memory 文件: {session.memory_file}')

        # 如果删除的是当前会话，切换到最新一个（如果有）
        if self._current_session_id == session_id:
            if self._sessions:
                self._current_session_id = list(self._sessions.keys())[-1]      #返回字典的所有键，取最后一个作为当前会话id
            else:
                self._current_session_id = None

        log_title(f'会话已删除: {session.name}')


    #@property装饰器，它将 current_session 方法伪装成一个属性，调用时不需要加括号，就像访问普通属性一样：self.current_session
    @property
    def current_session(self) -> Optional[Session]:
        if self._current_session_id:
            return self._sessions.get(self._current_session_id)
        return None


    #当前会话发送消息
    async def invoke(self, prompt: str) -> str:
        if not self.current_session:
            raise RuntimeError('当前没有活跃会话，请先调用 create_session() 或 switch_session()')
        return await self.current_session.invoke(prompt)

    #关闭所有会话的 MCP 连接
    async def close_all(self):
        for session in self._sessions.values():
            await session.close()
        log_title('所有会话已关闭')
