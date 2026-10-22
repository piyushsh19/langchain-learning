import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it in your shell environment "
            "or add it to a .env file."
        )

    # Initialize the OpenAI chat model.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create the DuckDuckGo search tool.
    search_tool = DuckDuckGoSearchRun()

    # Pull the standard ReAct prompt from LangChain Hub.
    prompt = hub.pull("hwchase17/react")

    # Create the ReAct agent with the search tool.
    agent = create_react_agent(
        llm=llm,
        tools=[search_tool],
        prompt=prompt,
    )

    # AgentExecutor runs the agent/tool loop.
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[search_tool],
        verbose=True,
        handle_parsing_errors=True,
    )

    question = input("Ask a question that may need web search: ").strip()
    if not question:
        question = "3 ways to reach Goa from Delhi"

    response = agent_executor.invoke({"input": question})
    print("\nAnswer:\n")
    print(response["output"])


if __name__ == "__main__":
    main()
