# Install if required:
# %pip install -U langchain-core pydantic

from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


# Step 1: Define the input schema using Pydantic
class MultiplyInput(BaseModel):
    a: int = Field(
        ..., description="The first number to multiply"
    )
    b: int = Field(
        ..., description="The second number to multiply"
    )


# Step 2: Create a custom tool by extending BaseTool
class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"
    args_schema: Type[BaseModel] = MultiplyInput

    # Step 3: Implement the tool logic
    def _run(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b


# Step 4: Create an instance of the tool
multiply_tool = MultiplyTool()


# Step 5: Invoke the tool
result = multiply_tool.invoke({"a": 3, "b": 5})
print("Result:", result)


# Step 6: Inspect tool metadata
print("Name:", multiply_tool.name)
print("Description:", multiply_tool.description)
print("Arguments:", multiply_tool.args)