from langchain_core.tools import tool

# Step 1: Create a custom tool using the @tool decorator
@tool
def calculate_total(price: float, tax_rate: float) -> float:
    """Calculate the total price including tax.

    Args:
        price: Original price before tax.
        tax_rate: Tax percentage to apply.
    """
    return price + (price * tax_rate / 100)


# Step 2: Invoke the tool directly
result = calculate_total.invoke({
    "price": 1000,
    "tax_rate": 18
})

print("Result:", result)

# Step 3: Inspect tool metadata
print("Tool name:", calculate_total.name)
print("Description:", calculate_total.description)
print("Arguments:", calculate_total.args)
print("schema:", calculate_total.args_schema.model_json_schema())