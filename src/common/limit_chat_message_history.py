from langchain_core.messages import BaseMessage
from langchain_core.chat_history import InMemoryChatMessageHistory,BaseChatMessageHistory
from pydantic import Field,BaseModel
import logging
logger = logging.getLogger(__file__)
class MessageHistory(InMemoryChatMessageHistory):
    max_token:int = Field(...)
    """
    扩展的聊天历史记录,可以限制聊天记录的最大长度
    """
    def add_message(self, message:BaseMessage):
        super().add_message(message)
        if(len(self.messages)>self.max_token):
            logger.warning("消息超限,立即压缩")
            self.messages = self.messages[-self.max_token]

from langchain.messages import AIMessage

class SessionHistory(BaseModel):
    max_token: int = Field(...)
    store: dict = Field(default_factory=dict)
    """
    处理历史消息
    """
    
    def process(self,session_id:str)->BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = MessageHistory(max_token=self.max_token)
        return self.store[session_id]
    
    def print_history(self,session_id:str):
        """
        查看消息记录
        """
        for message in self.store[session_id].messages:
            if isinstance(message, AIMessage):
                prefix = "AI"
            else:
                prefix = "User"
            print(f"{prefix}: {message.content}\n")


    