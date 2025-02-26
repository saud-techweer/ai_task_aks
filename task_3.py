import os
import re
from langchain_qdrant import Qdrant  
from langchain_huggingface import HuggingFaceEmbeddings  
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_core.documents import Document  

# Step 1: Set the File Path
file_path = os.path.join(os.path.dirname(__file__), "ironman.txt")

# Step 2: Read File and Split Paragraphs
def load_paragraphs(file_path):
    """Reads text file and splits paragraphs based on empty lines."""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    paragraphs = re.split(r"\n\s*\n", content.strip())  # Split by empty lines
    return [para.strip() for para in paragraphs if para.strip()]  # Remove empty items

paragraphs = load_paragraphs(file_path)
print(f"Loaded {len(paragraphs)} paragraphs from the file.")

# Step 3: Load Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Step 4: Determine Embedding Dimension Dynamically
sample_vector = embedding_model.embed_query("Test")
vector_dimension = len(sample_vector)

# Step 5: Connect to Qdrant and Create Collection
qdrant_client = QdrantClient("localhost", port=6333)
collection_name = "ironman_summary"

# If collection exists, delete and recreate it
if qdrant_client.collection_exists(collection_name):
    qdrant_client.delete_collection(collection_name)

qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_dimension, distance=Distance.COSINE)
)

# Step 6: Embed and Store Paragraphs in Qdrant
def store_paragraphs_in_qdrant(paragraphs):
    """Generates embeddings and stores paragraphs in Qdrant."""
    points = []
    for idx, para in enumerate(paragraphs):
        embedding_vector = embedding_model.embed_query(para)
        points.append(PointStruct(id=idx, vector=embedding_vector, payload={"page_content": para}))  # Fixed payload

    qdrant_client.upsert(collection_name=collection_name, points=points)
    print(f"Stored {len(paragraphs)} paragraphs in Qdrant.")

store_paragraphs_in_qdrant(paragraphs)

# Step 7: Use LangChain Ollama with `DeepSeek-R1:1.5B`
llm = OllamaLLM(model="deepseek-r1:1.5b")

# Step 8: Integrate Qdrant as a LangChain VectorStore
vector_store = Qdrant(
    client=qdrant_client,
    collection_name=collection_name,
    embeddings=embedding_model
)

retriever = vector_store.as_retriever()

# Step 9: Create LangChain RetrievalQA Pipeline
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# Step 10: Function to Ask LLM with Context
def ask_llm(query):
    """Retrieves relevant context and generates an LLM-powered response."""
    response = qa_chain.invoke({"query": query})  # Fixed deprecated `.run()`
    print("\n💡 **DeepSeek AI Response:**\n", response)

# Example Query
ask_llm("How did Tony Stark escape from captivity?")