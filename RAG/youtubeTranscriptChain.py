import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ------------------------------------------------------------------
# Environment Configuration
# ------------------------------------------------------------------
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# ------------------------------------------------------------------
# Step 1: Document Ingestion (YouTube Transcript)
# ------------------------------------------------------------------
video_id = "LPZh9BOjkQs"

try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    transcript_text = " ".join([chunk["text"] for chunk in transcript_list])
    print("Transcript retrieved successfully.")
except TranscriptsDisabled:
    raise RuntimeError("Captions are disabled for this video.")
except Exception as e:
    raise RuntimeError(f"Error fetching transcript: {e}")

# ------------------------------------------------------------------
# Step 2: Text Splitting & Indexing (Embeddings + Vector Store)
# ------------------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript_text])
print(f"Created {len(chunks)} chunks.")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)

# Configure retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# ------------------------------------------------------------------
# Step 3: Define Prompt, Model, and Helper Functions
# ------------------------------------------------------------------
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

prompt = PromptTemplate(
    template="""You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

Context:
{context}

Question: {question}
Answer:""",
    input_variables=["context", "question"]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
parser = StrOutputParser()

# ------------------------------------------------------------------
# Step 4: Construct and Execute the Chain
# ------------------------------------------------------------------
# Prepare inputs: retrieves and formats context, passes question unchanged
parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

# Full RAG LCEL Chain
main_chain = parallel_chain | prompt | llm | parser

# Run a query through the chain
query = "Can you summarize the main discussion in the video?"
response = main_chain.invoke(query)

print("\n--- Output ---")
print(response)