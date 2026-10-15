from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Define a sample long text
text = """LangChain is a framework for developing applications powered by language models. 
It enables applications that are context-aware and reason. 
Text splitters are a crucial part of the Retrieval-Augmented Generation (RAG) pipeline. 
They break large documents into smaller chunks so that an LLM can process them effectively."""

# 2. Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,       # Maximum number of characters per chunk
    chunk_overlap=20,     # Number of characters to overlap between chunks to keep context
    length_function=len,  # Function used to measure the chunk length
    separators=["\n\n", "\n", " ", ""] # Tries splitting by paragraphs, then lines, then spaces
)

# 3. Split the text into chunks
chunks = text_splitter.split_text(text)

# 4. View the output
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)
    print(f"(Length: {len(chunk)} characters)\n")