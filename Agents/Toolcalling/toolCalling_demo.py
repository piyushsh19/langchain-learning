from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def multiply(a: int, b: int) -> int:
    """Given 2 numbers a and b, this tool returns their product."""
    return a * b


def main() -> None:
    # Inspect the tool's metadata and input schema.
    print("Tool name:", multiply.name)
    print("Tool description:", multiply.description)
    print("Tool arguments:", multiply.args)
    print("Direct tool result:", multiply.invoke({"a": 3, "b": 4}))

    # Initialize a chat model. OPENAI_API_KEY must be set in the environment.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Bind the tool schema to the model so it can request tool calls.
    llm_with_tools = llm.bind_tools([multiply])

    # Ask a question that should lead the model to request the multiply tool.
    response = llm_with_tools.invoke(
        "What is 123 multiplied by 456? Use the multiply tool."
    )

    print("\nModel response:")
    print(response)

    # Tool binding lets the model request a tool call; it does not execute the
    # tool automatically. Execute any requested tool calls explicitly.
    for tool_call in response.tool_calls:
        if tool_call["name"] == multiply.name:
            result = multiply.invoke(tool_call["args"])
            print("\nTool call:", tool_call["name"])
            print("Arguments:", tool_call["args"])
            print("Tool result:", result)


if __name__ == "__main__":
    main()