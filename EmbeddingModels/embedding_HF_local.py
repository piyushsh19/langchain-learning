
# from langchain_huggingface import HuggingFaceEmbeddings

# embedding = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# text = "Delhi is the capital of India"

# vector = embedding.embed_query(text)

# print(vector)
# print("Embedding dimensions:", len(vector))

from langchain_huggingface import HuggingFaceEmbeddings

# Initialize the local embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
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

# Print each document and its embedding
for i, (document, vector) in enumerate(zip(documents, results), start=1):
    print(f"\nDocument {i}: {document}")
    print(f"Embedding: {vector}")
    print(f"Dimensions: {len(vector)}")

print(f"\nTotal documents embedded: {len(results)}")