import ollama


# -------------------------------------------------------------------
# Sample inventory database
# -------------------------------------------------------------------
inventory_db = {
    "laptop": {"stock": 5, "base_price": 1200},
    "monitor": {"stock": 0, "base_price": 300},
    "keyboard": {"stock": 25, "base_price": 80},
}


# -------------------------------------------------------------------
# Tool 1: Check inventory
# -------------------------------------------------------------------
def check_inventory(product_name: str) -> dict:
    """Get the current stock and base price for a product."""
    product_name = product_name.lower().strip()

    if product_name in inventory_db:
        return inventory_db[product_name]

    return {"stock": 0, "base_price": None}


# -------------------------------------------------------------------
# Tool 2: Calculate loyalty discount
# -------------------------------------------------------------------
def calculate_loyalty_discount(
    base_price: float, years_as_customer: int
) -> float: 
    """Calculate the final price using a loyalty discount."""
    discount = min(years_as_customer * 0.05, 0.30)  # Maximum discount: 30%
    final_price = base_price * (1 - discount)
    return round(final_price, 2)


# -------------------------------------------------------------------
# Map tool names to the actual Python functions
# -------------------------------------------------------------------
available_functions = {
    "check_inventory": check_inventory,
    "calculate_loyalty_discount": calculate_loyalty_discount,
}


# -------------------------------------------------------------------
# Tool schemas sent to Ollama
# -------------------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "check_inventory",
            "description": "Get stock and price for a product",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string"},
                },
                "required": ["product_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_loyalty_discount",
            "description": "Calculate final price based on loyalty years",
            "parameters": {
                "type": "object",
                "properties": {
                    "base_price": {"type": "number"},
                    "years_as_customer": {"type": "integer"},
                },
                "required": ["base_price", "years_as_customer"],
            },
        },
    },
]


def run_assistant(user_text: str) -> str:
    """Send a user request to Ollama, execute requested tools, and return a final answer."""
    messages = [{"role": "user", "content": user_text}]

    # Ask the model whether it needs to call a tool.
    response = ollama.chat(
        model="qwen3:8b",
        messages=messages,
        tools=tools,
    )

    # Check if the model requested any tools.
    tool_calls = response["message"].get("tool_calls")

    if tool_calls:
        for tool_call in tool_calls:
            function_info = tool_call["function"]
            tool_name = function_info["name"]
            tool_args = function_info["arguments"]

            # Find and execute the corresponding Python function.
            function_to_call = available_functions.get(tool_name)
            if function_to_call is None:
                result = {"error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    result = function_to_call(**tool_args)
                except (TypeError, ValueError) as exc:
                    result = {"error": str(exc)}

            # Add the model's tool request and the tool result to the conversation.
            messages.append(response["message"])
            messages.append(
                {
                    "role": "tool",
                    "content": str(result),
                }
            )

        # Ask the model to produce the final answer using the tool result.
        final_response = ollama.chat(
            model="qwen3:8b",
            messages=messages,
        )
        return final_response["message"]["content"]

    # If no tool was requested, return the model's response directly.
    return response["message"]["content"]


if __name__ == "__main__":
    print("Inventory and loyalty assistant (Ollama / qwen3:8b)")
    print("Example: 'I want to buy a laptop. Can you check stock?'")
    print("Example: 'I have been a customer for 5 years. What is the final price of a laptop?'")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        try:
            print(f"Assistant: {run_assistant(user_input)}\n")
        except Exception as exc:
            print(f"Error communicating with Ollama: {exc}\n")
