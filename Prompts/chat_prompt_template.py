from langchain_core.prompts import ChatPromptTemplate

# Step 1: Create a reusable chat prompt template
chat_template = ChatPromptTemplate([
    ("system", "You are a helpful {domain} expert."),
    ("human", "Explain in simple terms, what is {topic}?")
])

# Step 2: Provide values for the template variables
prompt = chat_template.invoke({
    "domain": "cricket",
    "topic": "Dusra"
})

# Step 3: Print the generated prompt messages
print(prompt)

# Step 4: Print the individual messages (optional)
print("\nIndividual messages:")

for message in prompt.messages:
    print(f"{message.type}: {message.content}")