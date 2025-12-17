# pip install pypdf
import os


def get_file_path():
    """获取文件路径"""
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    file_path = os.path.join(current_dir,"..","assert","2505.20829v2.pdf")
    return file_path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_file_content(file_path:str)->list[Document]:
    pdf_loader = PyPDFLoader(file_path)
    docs = pdf_loader.load()
    print(f'加载文件成功,成功加载{len(docs)}')
    # print (type(docs[0]))
    # print (docs[0])
    return docs

from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_docs(docs:list[Document])->list[Document]:
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=1000,chunk_overlap=200,add_start_index=True
    )
    all_splits = text_spliter.split_documents(docs)
    return all_splits

from langchain_ollama import OllamaEmbeddings

embending = OllamaEmbeddings(model="qwen3-embedding:4b")

from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embending)

def get_vertor_store() -> InMemoryVectorStore:
    file_path = get_file_path()
    file_contents = load_file_content(file_path)
    all_splits = split_docs(file_contents)
    vector_store.add_documents(documents=all_splits)
    return vector_store

def similarity_search(query:str)->list[Document]:
    """内存矢量数据库检索测试"""
    return vector_store.similarity_search(query=query)

def similarity_search_by_score(query:str)->list[tuple[Document,float]]:
    """内存矢量数据库检索测试
    返回文档评分，分数越高，文档越相似。
    """
    return vector_store.similarity_search_with_score(query=query)

def embending_query(query:str)->list[Document]:
    """嵌入查询测试"""
    query_result = embending.embed_query(query)
    results = vector_store.similarity_search_by_vector(query_result)
    return results
from langchain_core.runnables import RunnableLambda

def retriever(query: str) -> list[Document]:
    """检索单个查询的文档"""
    return vector_store.similarity_search(query, k=1)

def retriever_batch_1(queries:str):
    retriever_runnable = RunnableLambda(retriever)
    return retriever_runnable.batch(queries)

def retriever_batch_2(queries:list[str]):
    retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":1})
    return retriever.batch(queries)

if __name__ == '__main__':
    get_vertor_store()
    # results = similarity_search("What imitations dose Force-aware have?")
    # print(f'similarity_search results[0]:\n{results[0]}')

    # result2 = similarity_search_by_score("What imitations dose Force-aware have?")
    # doc,score = result2[0]
    # print(f"Score and doc: {score}\n{doc}")


    # results3 = embending_query("What imitations dose Force-aware have?")
    # print(f'embed_query results[0]:\n{results3[0]}') 

    query = [
        "What imitations dose Force-aware have?",
        "What was Force-aware Imitation Learning?",
    ]

    results = retriever_batch_1(query)
    print(f'retriever.batch 1:\n{results}')
    results = retriever_batch_2(query)
    print(f'retriever.batch 2:\n{results}')

