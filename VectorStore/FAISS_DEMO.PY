from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1. Initialize the embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Create sample documents
docs = [
    Document(page_content="Apples and oranges are both delicious fruits.", metadata={"category": "food"}),
    Document(page_content="The new electric car has a range of 400 miles.", metadata={"category": "auto"}),
    Document(page_content="Bananas are a great source of potassium.", metadata={"category": "food"})
]

# 3. Create the FAISS vector store in-memory
vectorstore = FAISS.from_documents(docs, embeddings)

# 4. Perform a similarity search
query = "Tell me about healthy snacks."
results = vectorstore.similarity_search(query, k=2)

print("--- FAISS Search Results ---")
for i, result in enumerate(results):
    print(f"Result {i+1}: {result.page_content} (Metadata: {result.metadata})")

# 5. Save and Load FAISS database locally (Optional but highly recommended)
vectorstore.save_local("faiss_index")
loaded_vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)