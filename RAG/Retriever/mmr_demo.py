from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
texts = [
    "LangChain is a framework for developing LLM applications.",
    "LangChain makes it easy to build RAG pipelines.",
    "Python is a popular programming language.",
    "The weather is sunny today."
]
vectorstore = Chroma.from_texts(texts, embeddings)

# 2. Convert the vector store into a retriever
# search_type="mmr" enables diverse results
# k=2 returns the top 2 final documents
retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 2}
)

# 3. Retrieve documents
query = "What is LangChain?"
results = retriever.invoke(query)

print("--- Vector Store (MMR) Results ---")
for doc in results:
    print(doc.page_content)