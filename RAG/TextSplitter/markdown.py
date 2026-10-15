from langchain_text_splitters import MarkdownHeaderTextSplitter

# 1. Define a markdown document
markdown_document = """
# Introduction
Welcome to the documentation. 

## Setup
To get started, install the package.

## Usage
Run the main script to start the server.
"""

# 2. Define the headers to split on and their metadata keys
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]

# 3. Initialize the splitter
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# 4. Split the document
md_header_splits = markdown_splitter.split_text(markdown_document)

# 5. View the chunks and their structural metadata
for split in md_header_splits:
    print(f"Content: {split.page_content}")
    print(f"Metadata: {split.metadata}\n")