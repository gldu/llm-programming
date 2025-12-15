
from langchain.chat_models import init_chat_model
def get_llm():
    return init_chat_model(
        model="ollama:deepseek-r1:1.5b",
        temperature=0.3,
        base_url="http://localhost:11434",
        verbose=True)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from common.limit_chat_message_history import SessionHistory

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.messages import HumanMessage

session_history = SessionHistory(max_token=20)

def add_session_message(session_id:str)->ChatMessageHistory:
    return session_history.process(session_id)

def get_history_chain():
    prompt = ChatPromptTemplate.from_messages([
        (
            "system","You are a helpful assistant. You can answer all questions to the best of your ability."
        ),MessagesPlaceholder(variable_name="message")
    ])
    chain = prompt | get_llm()
    return RunnableWithMessageHistory(chain,add_session_message)

with_messages_chain = get_history_chain()

def chat(hunman_message,session_id):
    response = with_messages_chain.invoke(
        [HumanMessage(content=hunman_message)],
                            config={"configurable":{"session_id":session_id}})
    return response.content

if __name__ == "__main__":
    session_id = "liu123"
    print (chat("你知道x-space的老板马斯克么？", session_id))
    print (chat("他出生在哪个国家？", session_id))
    print (chat("他和特朗普是什么关系？", session_id))
  
    session_history.print_history(session_id)





