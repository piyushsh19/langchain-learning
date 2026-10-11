from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load variables from the project-root .env file
load_dotenv()

# Initialize the chat model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Send a prompt
response = llm.invoke("Explain LangChain in simple terms.")

# Print the response
print(response.content)