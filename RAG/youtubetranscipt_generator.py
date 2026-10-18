import streamlit as st
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------
st.set_page_config(page_title="YouTube RAG Assistant", page_icon="📺", layout="wide")
st.title("📺 YouTube Video Assistant")
st.markdown("Chat with any YouTube video using LangChain, FAISS, and OpenAI.")

# ------------------------------------------------------------------
# Session State Management
# ------------------------------------------------------------------
# We use session state so the database isn't rebuilt every time you ask a question
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------------------------------------------------
# Sidebar Configuration
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    video_input = st.text_input("YouTube Video ID or URL", placeholder="e.g., LPZh9BOjkQs")
    process_btn = st.button("Process Video", type="primary", use_container_width=True)
    
    st.divider()
    st.caption("This app creates a temporary FAISS vector store in memory. Refreshing the page clears the data.")

# ------------------------------------------------------------------
# Processing Logic
# ------------------------------------------------------------------
if process_btn:
    if not api_key:
        st.error("Please provide an OpenAI API Key.")
    elif not video_input:
        st.error("Please provide a YouTube Video ID or URL.")
    else:
        # Extract Video ID if a full URL was provided
        if "v=" in video_input:
            video_id = video_input.split("v=")[-1].split("&")[0]
        elif "youtu.be/" in video_input:
            video_id = video_input.split("youtu.be/")[-1].split("?")[0]
        else:
            video_id = video_input
            
        os.environ["OPENAI_API_KEY"] = api_key
        
        with st.spinner(f"Fetching transcript for '{video_id}' and building vector database..."):
            try:
                # 1. Ingestion
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
                transcript_text = " ".join([chunk["text"] for chunk in transcript_list])
                
                # 2. Splitting & Indexing
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.create_documents([transcript_text])
                
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                vector_store = FAISS.from_documents(chunks, embeddings)
                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
                
                # 3. Prompt & Model Setup
                def format_docs(retrieved_docs):
                    return "\n\n".join(doc.page_content for doc in retrieved_docs)

                prompt = PromptTemplate(
                    template="""You are a helpful assistant. Answer ONLY from the provided transcript context.
                    If the context is insufficient, just say you don't know.
                    
                    Context:
                    {context}
                    
                    Question: {question}
                    Answer:""",
                    input_variables=["context", "question"]
                )
                
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
                parser = StrOutputParser()

                # 4. Create the Chain
                parallel_chain = RunnableParallel({
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough()
                })
                
                # Save the complete chain into Streamlit's session state
                st.session_state.rag_chain = parallel_chain | prompt | llm | parser
                st.session_state.messages = [] # Clear previous chat history
                
                st.success("✅ Video processed successfully! You can now ask questions.")

            except TranscriptsDisabled:
                st.error("❌ Captions are disabled for this video.")
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")

# ------------------------------------------------------------------
# Main Chat Interface
# ------------------------------------------------------------------
# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input Trigger
if user_query := st.chat_input("Ask a question about the video..."):
    if st.session_state.rag_chain is None:
        st.warning("⚠️ Please configure your API key and process a video first!")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing transcript..."):
                response = st.session_state.rag_chain.invoke(user_query)
                st.markdown(response)
        
        # Save assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})