from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# 1. Define the Python code as a string
text = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade  # Grade is a float (like 8.5 or 9.2)

    def get_details(self):
        return self.name"""

# 2. Initialize the splitter specifically for Python code
# This tells LangChain to try splitting by classes, then functions, then standard characters
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, 
    chunk_size=60, 
    chunk_overlap=0
)

# 3. Split the code into chunks
docs = python_splitter.create_documents([text])

# 4. View the output
for i, doc in enumerate(docs):
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content)
    print(f"(Length: {len(doc.page_content)})\n")