
from pydantic import BaseModel, Field
from typing import Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import HuggingFacePipeline


# 1. Define the Pydantic schema
class Person(BaseModel):
    name: str = Field(description="Full name of the fictional person")
    age: int = Field(description="Age of the person", ge=1, le=120)
    city: str = Field(description="Fictional city where the person lives")


# 2. Create the output parser
parser = PydanticOutputParser(pydantic_object=Person)


# 3. Create the prompt template
template = PromptTemplate(
    template=(
        "Generate the name, age, and city of a fictional person "
        "inspired by {place}.\n"
        "The city must be fictional, not a real-world city.\n"
        "Return only valid JSON matching these instructions:\n"
        "{format_instructions}"
    ),
    input_variables=["place"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)


# 4. Load the local TinyLlama model
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    model_kwargs={"local_files_only": True},
    pipeline_kwargs={
        "max_new_tokens": 200,
        "do_sample": False,
        "return_full_text": False,
    },
)


# 5. Generate the prompt
prompt = template.invoke({"place": "Sri Lankan culture"})

print("\n--- Generated Prompt ---")
print(prompt.to_string())


# 6. Get the model response
response = llm.invoke(prompt.to_string())

print("\n--- Raw Model Output ---")
print(response)


# 7. Parse the response into a validated Pydantic object
try:
    person = parser.parse(response)

    print("\n--- Parsed Person ---")
    print(person)

    print("\n--- Individual Fields ---")
    print("Name:", person.name)
    print("Age:", person.age)
    print("City:", person.city)

    print("\n--- JSON Output ---")
    print(person.model_dump_json(indent=2))

except Exception as error:
    print("\nCould not parse the model response:", error)
