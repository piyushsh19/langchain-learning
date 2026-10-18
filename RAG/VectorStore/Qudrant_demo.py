from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1. Initialize the embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Create sample documents
docs = [
    Document(page_content="Machine learning allows systems to learn from data.", metadata={"topic": "AI"}),
    Document(page_content="Photosynthesis is how plants convert sunlight into energy.", metadata={"topic": "biology"}),
    Document(page_content="Neural networks are inspired by the human brain.", metadata={"topic": "AI"})
]

# 3. Create the Qdrant vector store in-memory
# location=":memory:" ensures it runs completely in RAM without a separate server
vectorstore = QdrantVectorStore.from_documents(
    docs,
    embeddings,
    location=":memory:", 
    collection_name="my_documents",
)

# 4. Perform a similarity search
query = "How does artificial intelligence work?"
results = vectorstore.similarity_search(query, k=2)

print("--- Qdrant Search Results ---")
for i, result in enumerate(results):
    print(f"Result {i+1}: {result.page_content} (Metadata: {result.metadata})")