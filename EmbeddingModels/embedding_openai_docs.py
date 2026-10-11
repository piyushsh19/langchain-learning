
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize the embedding model
embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=32,
)

# Multiple documents
documents = [
    "Delhi is the capital of India.",
    "Mumbai is the financial capital of India.",
    "Python is a popular programming language.",
    "LangChain is a framework for building LLM applications.",
]

# Generate embeddings for all documents
results = embedding.embed_documents(documents)

# Display results
for i, (document, vector) in enumerate(zip(documents, results), start=1):
    print(f"\nDocument {i}: {document}")
    print(f"Embedding: {vector}")
    print(f"Dimensions: {len(vector)}")
