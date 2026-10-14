from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json

load_dotenv()

model = ChatOpenAI()

review_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "sentiment": {
            "type": "string",
            "enum": ["pos", "neg"]
        },
        "key_themes": {
            "type": "array",
            "items": {"type": "string"}
        },
        "pros": {
            "anyOf": [
                {"type": "array", "items": {"type": "string"}},
                {"type": "null"}
            ]
        },
        "cons": {
            "anyOf": [
                {"type": "array", "items": {"type": "string"}},
                {"type": "null"}
            ]
        },
        "name": {
            "anyOf": [
                {"type": "string"},
                {"type": "null"}
            ]
        }
    },
    "required": ["summary", "sentiment", "key_themes"]
}

structured_model = model.with_structured_output(review_schema)

result = structured_model.invoke("""
The hardware is great, but the software feels bloated.
There are too many pre-installed apps that I can't remove.
The UI also looks outdated.
""")

print(json.dumps(result, indent=2))