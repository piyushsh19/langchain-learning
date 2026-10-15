from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Define text with clear topic shifts
text = """
The James Webb Space Telescope is the largest optical telescope in space. It is designed to conduct infrared astronomy. Its high resolution and sensitivity allow it to view objects too old, distant, or faint for the Hubble Space Telescope.

Meanwhile, baking a chocolate cake requires precise measurements. You need flour, sugar, cocoa powder, baking soda, and eggs. Mix the dry ingredients first, then slowly fold in the wet ingredients to ensure a moist texture.

In sports news, the local football team won their championship match last night. The star quarterback threw for 300 yards and three touchdowns. Fans flooded the streets to celebrate the historic victory.
"""

# 2. Initialize an embedding model (this converts text into mathematical meaning)
# We use a fast, local Hugging Face model here so you don't need an API key
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Initialize the Semantic Chunker
# It uses the embeddings to compare sentences. If two sentences are too different in meaning, it splits them.
semantic_splitter = SemanticChunker(embeddings)

# 4. Split the text
docs = semantic_splitter.create_documents([text])

# 5. View the output
for i, doc in enumerate(docs):
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content.strip())
    print()