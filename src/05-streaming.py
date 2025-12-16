import random
from langchain.tools import tool


@tool
def where_cat_is_hiding() -> str:
    """Where is the cat hiding right now?"""
    return random.choice(["under the bed", "on the shelf"])

@tool
def get_items(place: str) -> str:
    """Use this tool to look up which items are in the given place."""
    if "bed" in place:  # For under the bed
        return "socks, shoes and dust bunnies"
    if "shelf" in place:  # For 'shelf'
        return "books, penciles and pictures"
    else:  # if the agent decides to ask about a different place
        return "cat snacks"
    
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system","you are a helpful assistant."),
    ("placeholder","{chat_history}"),
    ("human","{human_message}"),
    ("placeholder","{agent_scratchpad}")
])

tools = [where_cat_is_hiding,get_items]

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm_model_name = "qwen3:4b"

class CustomStreamingHandler(StreamingStdOutCallbackHandler):
    def __init__(self):
        super().__init__()
        self.buffer = []
    
    def on_llm_new_token(self, token, **kwargs):
        self.buffer.append(token)
        print(token, end="^", flush=True)  # 直接流式输出最终结果
model = ChatOllama(model=llm_model_name,temperature=0.3,verbose=True,callbacks=[StreamingStdOutCallbackHandler()])

agent = create_agent(
            model=model,
            tools=tools,
            name="Agent"
            )
def ask_message(question):
   for event in agent.stream({"messages":[{"role":"user","content":question}]},stream_mode="messages"):
      print(event[0].content)

def ask_values(question):
   for event in agent.stream({"messages":[{"role":"user","content":question}]},stream_mode="values"):
       messages = event["messages"]
       for message in messages:
         message.pretty_print()

if __name__ == '__main__':
    ask_message("what's items are located where the cat is hiding?")
    # ask_values("what's items are located where the cat is hiding?")




