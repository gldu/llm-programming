import os

from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document

def get_assert_path()->str:
   current_file_path =  os.path.abspath(__file__)
   current_dir = os.path.dirname(current_file_path)
   file_path = os.path.join(current_dir,"..","assert","law.csv")
   return file_path

def get_db_path()->str:
   current_file_path =  os.path.abspath(__file__)
   current_dir = os.path.dirname(current_file_path)
   file_path = os.path.join(current_dir,"..","storage",'db_law')
   return file_path

embending = OllamaEmbeddings(model="qwen3-embedding:4b")

vector_db = Chroma(
       persist_directory=get_db_path(),
       collection_name="csv_collection",
       embedding_function=embending
       )

def embed_documents_in_batches(documents,batch_size:int=10):
    """
    按批次嵌入，可以显示进度。
    vectordb会自动持久化存储在磁盘。
    """
    for i in tqdm(range(0, len(documents), batch_size), desc="嵌入进度"):
        batch = documents[i:i + batch_size]
        vector_db.add_documents(batch)

def create():
    """对文本矢量化并存储在本地磁盘"""
    loader = CSVLoader(get_assert_path(),csv_args={"delimiter":"#"},autodetect_encoding=True)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embed_documents_in_batches(texts,batch_size=3)

def search(query:str)->list[tuple[Document,float]]:
    return vector_db.similarity_search_with_score(query,k=1)

if __name__ == '__main__':
    create()
    results = search("恶意商标申请")
    print(f'search results:\n{results}')
 




