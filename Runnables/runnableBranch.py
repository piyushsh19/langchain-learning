import os
import transformers

# Suppress Hugging Face warnings and loading bars
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
transformers.logging.set_verbosity_error()
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableBranch,RunnablePassthrough,RunnableLambda

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

def word_count(text):
    return len(text.split())

prompt_template = PromptTemplate(
    template='write a detail report about \n {topic}',
    input_variables=['topic']
)

prompt_template2 = PromptTemplate(
    template='summarize the follow text \n {text}',
    input_variables=['text']
)
parser =StrOutputParser()

report_gen_chain = RunnableSequence (prompt_template,model,parser)

branch_chain = RunnableBranch (
    (lambda x : len(x.split()) >500 ,RunnableSequence(prompt_template2,model,parser)),
    RunnablePassthrough()
)
final_chain = RunnableSequence(report_gen_chain,branch_chain)

result = final_chain.invoke({'topic':'russia vs ukrain'})
print(result)