from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1. Create sample documents with metadata
documents = [
    Document(page_content="The James Webb Space Telescope conducts infrared astronomy.", metadata={"topic": "space", "year": 2021}),
    Document(page_content="Baking a chocolate cake requires flour, sugar, and cocoa.", metadata={"topic": "cooking", "difficulty": "easy"}),
    Document(page_content="The local football team won their championship match last night.", metadata={"topic": "sports", "league": "local"}),
    Document(page_content="The Hubble Space Telescope was launched into low Earth orbit in 1990.", metadata={"topic": "space", "year": 1990})
]

# 2. Initialize a local embedding model
# This converts our text into numerical vectors so Chroma can measure their semantic similarity
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Initialize the Chroma Vector Store and ingest the documents
# We are storing this in-memory for the demo. To save to disk, add: persist_directory="./chroma_db"
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
)

# 4. Define a search query
query = "Tell me about space exploration and orbital telescopes."
print(f"Search Query: '{query}'\n")

# 5. Perform a similarity search
# k=2 tells Chroma to return the 2 most semantically relevant documents
results = vectorstore.similarity_search(query, k=2)
# results = vectorstore.similarity_search_with_score(query, k=2)

# 6. Display the retrieved documents
for i, result in enumerate(results):
    print(f"--- Top Result {i+1} ---")
    print(f"Content: {result.page_content}")
    print(f"Metadata: {result.metadata}\n")