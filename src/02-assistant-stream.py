from langchain.chat_models import init_chat_model
from common.limit_chat_message_history import SessionHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.messages import  HumanMessage
def get_llm():
    return init_chat_model(
        model="ollama:deepseek-r1:1.5b",
        base_url="http://localhost:11434",
        temperature=0,
        verbose=True
    )

session_history = SessionHistory(max_token=20)

def get_message_history(session_id:str):
   return session_history.process(session_id)

def get_history_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system","You are a helpful assistant. You can answer all questions to the best of your ability in {language}."),
        MessagesPlaceholder(variable_name="messages")
    ])
    chat_chain = prompt | get_llm()
    return RunnableWithMessageHistory(chat_chain, get_message_history, input_messages_key="messages")

with_message_chain = get_history_chain()

def chat(hunman_message:str,session_id:str,language:str="English"):
    for chunk in with_message_chain.stream(
        {"messages":[HumanMessage(content=hunman_message)],"language":language},
        config={"configurable":{"session_id":session_id}},
    ):
        print(chunk.content,end="",flush=True)
    print()

if __name__ == '__main__':
    session_id = "aaaa"

    chat("你知道x-space的老板马斯克么？",session_id,language="简体中文")
    chat("他出生在哪个国家？",session_id)
 
