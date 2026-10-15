from langchain_text_splitters import CharacterTextSplitter

# 1. Define a sample text
text = "LangChain is a framework for developing applications powered by language models. It enables applications that are context-aware and reason. Text splitters are a crucial part of the RAG pipeline. They break large documents into smaller chunks."

# 2. Initialize the Character text splitter
text_splitter = CharacterTextSplitter(
    separator=". ",       # The specific character(s) to split on (e.g., end of a sentence)
    chunk_size=60,        # Maximum number of characters per chunk
    chunk_overlap=0,      # Number of characters to overlap between chunks
    length_function=len
)

# 3. Split the text into chunks
chunks = text_splitter.split_text(text)

# 4. View the output
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)
    print(f"(Length: {len(chunk)} characters)\n")