from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader

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

loader = PyPDFLoader('ai-engineering.pdf')
dict_loader = DirectoryLoader (
    path = 'books',
    glob = '*.pdf',
    loader_cls= PyPDFLoader
)

docs =loader.load()

docs_dict = dict_loader.lazy_load()

for  document in docs_dict:
    print(document.metadata)


print(len(docs))

print(len(docs_dict ))