from langchain_ollama import ChatOllama
from common.limit_chat_message_history import SessionHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
def get_llm():
    return ChatOllama(
        model="deepseek-r1:1.5b",
        base_url="http://localhost:11434",
        temperature=0.4,
        verbose=True)

session_history = SessionHistory(max_token=40)

def get_session_history(session_id:str):
    return session_history.process(session_id)

def get_chain_history():
    chain = get_llm() | StrOutputParser()
    return RunnableWithMessageHistory(chain,get_session_history)

with_message_history = get_chain_history()

def chat(hunman_message:str,session_id:str):
    return with_message_history.invoke(
        [HumanMessage(content=hunman_message)],
        config={"configurable":{"session_id":session_id}})


if __name__ == "__main__":
    session_id = "aaaa"
    print (chat("计算 5 加 3 乘以 2 的结果是多少？", session_id))

    print (chat("你知道x-space的老板马斯克么？", session_id))
    print (chat("他出生在哪个国家？", session_id))
    print (chat("他和特朗普是什么关系？", session_id))
