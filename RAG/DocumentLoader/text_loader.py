from langchain_community.document_loaders import TextLoader

import os
import transformers

# Suppress Hugging Face warnings and loading bars
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
transformers.logging.set_verbosity_error()
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Step 1: Load the local Hugging Face model (with warning fixes)
model = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-3B-Instruct",
    task="text-generation",
    model_kwargs={"local_files_only": False},
    pipeline_kwargs={
        "max_new_tokens": 400,
        "max_length": None,
        "do_sample": False,
        "return_full_text": False,
        "clean_up_tokenization_spaces": False
    },
)
prompt_template = PromptTemplate(
    template='write a summary  about \n {topic}',
    input_variables=['topic']
)

parser =StrOutputParser()

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

print(docs)

print(type(docs))

print(len(docs))


print(type(docs[0]))
print(docs[0].metadata)

print('-------------------------summary -----------------')

chain = prompt_template | model | parser 

result = chain.invoke({'topic': docs[0].page_content})

print(result)