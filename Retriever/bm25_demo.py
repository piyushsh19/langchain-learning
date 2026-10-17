from langchain_community.retrievers import BM25Retriever

# 1. Define sample texts
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast brown fox leaps over a sleeping dog.",
    "The dog is lazy and sleeping in the sun."
]

# 2. Initialize the BM25 Retriever directly from texts
# k=2 returns the top 2 documents based on keyword frequency
bm25_retriever = BM25Retriever.from_texts(texts, k=2)

# 3. Retrieve documents
# It will look for exact matches for the words "lazy" and "dog"
query = "lazy dog"
results = bm25_retriever.invoke(query)

print("--- BM25 (Keyword) Results ---")
for doc in results:
    print(doc.page_content)