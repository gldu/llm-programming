from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

def get_llm():
    return OllamaLLM(
        model="deepseek-r1:1.5b",
        temperature=0.3,
        base_url="http://localhost:11434",
        verbose=True)

def ask(question:str):
    tempalte = """Question: {question}"""
    prompt = ChatPromptTemplate.from_template(tempalte)
    chain = prompt | get_llm()
    result = chain.invoke({"question":question})
    return result

if __name__ == '__main__':
    print(ask("你是谁？你有什么本事？"))