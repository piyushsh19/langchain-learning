import os
import transformers

# Suppress Hugging Face warnings and loading bars
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
transformers.logging.set_verbosity_error()
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence ,RunnableParallel

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
    template='Genrate a tweet about \n {topic}',
    input_variables=['topic']
)

prompt_template2 = PromptTemplate(
    template='Generate the linkedin post about\n {topic}',
    input_variables=['topic']
)
parser = StrOutputParser()

parallel_parser = RunnableParallel({
    'tweet': RunnableSequence(prompt_template,model,parser),
    'linkedin': RunnableSequence(prompt_template2,model,parser)
})

result = parallel_parser.invoke({'topic':'ai'})

print(result)