from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent

from langchain_openai import ChatOpenAI

# 1. Create the search tool
search_tool = DuckDuckGoSearchRun()

# 2. Initialize the LLM
llm = ChatOpenAI(model="gpt-4.1-mini")

# 3. Create the agent and give it the tool
agent = create_agent(
    model=llm,
    tools=[search_tool],
    system_prompt=(
        "You are a helpful research assistant. "
        "Use the search tool when current or external "
        "information is needed. Summarize results clearly."
    ),
)

# 4. Ask a question
response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Search for LangChain agents and explain them."
        }
    ]
})

# 5. Print the final answer
print(response["messages"][-1].content)