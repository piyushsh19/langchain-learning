# Install if required:
# %pip install -U langchain-core

from langchain_core.tools import tool


# Step 1: Create the addition tool
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# Step 2: Create the multiplication tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# Step 3: Create a toolkit to group the tools
class MathToolkit:
    def get_tools(self):
        return [add, multiply]


# Step 4: Initialize the toolkit and retrieve tools
toolkit = MathToolkit()
tools = toolkit.get_tools()


# Step 5: Display tool information
for tool in tools:
    print(tool.name, "=>", tool.description)


# Step 6: Execute the tools directly
print("Addition result:", add.invoke({"a": 10, "b": 5}))
print("Multiplication result:", multiply.invoke({"a": 10, "b": 5}))