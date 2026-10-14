from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, Literal, Annotated

load_dotenv()

model = ChatOpenAI()


class Review(BaseModel):
    key_themes: list[str] = Field(
        description="The main themes discussed in the review"
    )

    summary: str = Field(
        description="A brief summary of the review"
    )

    sentiment: Literal["pos", "neg"] = Field(
        description="The sentiment of the review: positive or negative"
    )

    pros: Optional[list[str]] = Field(
        default=None,
        description="All the advantages mentioned in the review"
    )

    cons: Optional[list[str]] = Field(
        default=None,
        description="All the disadvantages mentioned in the review"
    )

    name: Optional[str] = Field(
        default=None,
        description="The product name, if mentioned in the review"
    )


structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
The hardware is great, but the software feels bloated.
There are too many pre-installed apps that I can't remove.
Also, the UI looks outdated compared to other brands.
Hoping for a software update to fix this.
""")

print(result)
print(result.summary)
print(result.sentiment)
print(result.key_themes)
print(result.pros)
print(result.cons)
print(result.name)