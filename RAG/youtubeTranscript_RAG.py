import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# ==========================================
# Step 1a - Indexing (Document Ingestion)
# ==========================================
video_id = "LPZh9BOjkQs" # only the ID, not full URL
transcript_text = ""

try:
    # If you don't care which language, this returns the "best" one
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    
    # Flatten it to plain text
    transcript_text = " ".join([chunk["text"] for chunk in transcript_list])
    print("Transcript downloaded successfully.")

except TranscriptsDisabled:
    print("No captions available for this video.")
    exit()
except Exception as e:
    print(f"An error occurred fetching transcript: {e}")
    exit()

# ==========================================
# Step 1b - Indexing (Text Splitting)
# ==========================================
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# We pass the string wrapped in a list to create_documents
chunks = splitter.create_documents([transcript_text])
print(f"Split transcript into {len(chunks)} chunks.")

# ==========================================
# Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)
# ==========================================
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)
print("Vector store created successfully.")

# ==========================================
# Step 2 - Retrieval
# ==========================================
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

question = "is the topic of aliens discussed in this video? if yes then what was discussed"
retrieved_docs = retriever.invoke(question)

# Combine retrieved documents into a single context string
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

# ==========================================
# Step 3 - Prompt Setup
# ==========================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    {context}
    Question: {question}
    """,
    input_variables=['context', 'question']
)

# Format the prompt with the retrieved context and question
final_prompt = prompt.invoke({"context": context_text, "question": question})

# ==========================================
# Step 4 - Generation
# ==========================================
answer = llm.invoke(final_prompt)

print("\n--- Final Answer ---")
print(answer.content)