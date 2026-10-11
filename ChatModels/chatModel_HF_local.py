
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFacePipeline,
)

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 1000,
        "do_sample": True,
        "temperature": 0.5,
        "return_full_text": False,
    },
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("Meditation exercise?")
print(response.content)