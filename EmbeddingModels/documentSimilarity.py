from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the local embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Documents to compare
documents = [
    "Virat Kohli is an Indian international cricketer and former captain.",
    "Virat Kohli is known for his batting records and aggressive playing style.",
    "MS Dhoni is a former Indian cricket captain and wicketkeeper.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

# User query
query = "Tell me about Bumrah"

# Generate document and query embeddings
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# Calculate cosine similarity
scores = cosine_similarity(
    [query_embedding],
    doc_embeddings
)[0]

# Find the document with the highest similarity score
index, score = sorted(
    enumerate(scores),
    key=lambda x: x[1]
)[-1]

# Print the result
print("Query:", query)
print("Most relevant document:", documents[index])
print("Similarity score:", round(float(score), 4))

