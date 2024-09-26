from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from dotenv import load_dotenv

print("loading env")
load_dotenv()

print("loading llm")
llm = Ollama(model="gemma2:2b", request_timeout=120.0)

print("parser")
parser = LlamaParse(result_type="markdown")

file_extractor = {".pdf":parser}
documents = SimpleDirectoryReader("./docs", file_extractor=file_extractor).load_data()

print("converting into embeddings")
embed_model = resolve_embed_model("local:BAAI/bge-m3")
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)


print("loading engine")
query_eng = vector_index.as_query_engine(llm=llm)

while True:
    try:
        print(query_eng.query(input("Query: ")))
    
    except Exception as e:
        print(e)
    
    
    