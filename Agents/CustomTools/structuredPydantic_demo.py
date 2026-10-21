# Install if required:
# %pip install -U langchain langchain-core pydantic

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field


# Step 1: Define the Pydantic input schema
class MultiplyInput(BaseModel):
    a: int = Field(
        ..., description="The first number to multiply"
    )
    b: int = Field(
        ..., description="The second number to multiply"
    )


# Step 2: Create the Python function
def multiply_func(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# Step 3: Convert the function into a LangChain tool
multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput
)


# Step 4: Invoke the tool
result = multiply_tool.invoke({"a": 3, "b": 5})
print("Result:", result)


# Step 5: Inspect the tool
print("Name:", multiply_tool.name)
print("Description:", multiply_tool.description)
print("Arguments:", multiply_tool.args)