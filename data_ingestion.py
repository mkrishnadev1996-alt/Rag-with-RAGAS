from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
import os
from asyncio import run
from dotenv import load_dotenv

load_dotenv()

embeddings = PineconeEmbeddings(model="llama-text-embed-v2")

if not os.environ.get("PINECONE_API_KEY"):
    raise ValueError("PINECONE_API_KEY not set")

# vector store
vector_store = PineconeVectorStore(
    embedding=embeddings,
    index_name="rag-with-ragas",
    host="https://rag-with-ragas-jaff7kv.svc.aped-4627-b74a.pinecone.io"
)


# retriever
retriever  = vector_store.as_retriever(search_kwargs={"k":6})

# load PDF doc, chunk, embed and add to vector store
async def load_data():
    # load data from pdf
    loader = PyMuPDFLoader(file_path="data/llama2.pdf")
    docs = loader.load()

    # Split to chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    doc_chunks = splitter.split_documents(docs)

    print(f"Original docs: {len(docs)}")
    print(f"Chunks: {len(doc_chunks)}")

    # Async add docs
    print("Adding Docs to vector store")
    await vector_store.aadd_documents(documents=doc_chunks,batchsize=20)
    print("Docs added to vector store")
    

if __name__=="__main__":
    run(load_data())