import os
import transformers

# Suppress Hugging Face warnings and loading bars
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
transformers.logging.set_verbosity_error()
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

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

# Parsers
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive','negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Step 2: Classifier Chain (with ChatML formatting for Qwen to prevent parsing errors)
prompt1 = PromptTemplate(
    template="""<|im_start|>system
You are a helpful assistant. You must output strictly valid JSON and nothing else.
<|im_end|>
<|im_start|>user
Classify the sentiment of the following feedback text into positive or negative.

Feedback: {feedback}

{format_instruction}
<|im_end|>
<|im_start|>assistant
""",
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

# Step 3: Response Prompts
prompt2 = PromptTemplate(
    template='<|im_start|>user\nWrite an appropriate response to this positive feedback: \n {feedback}<|im_end|>\n<|im_start|>assistant\n',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='<|im_start|>user\nWrite an appropriate response to this negative feedback: \n {feedback}<|im_end|>\n<|im_start|>assistant\n',
    input_variables=['feedback']
)

# Step 4: Branching Logic
# We check x['sentiment'].sentiment because we will pass a dict containing the parsed Pydantic object
branch_chain = RunnableBranch(
    (lambda x: x['sentiment'].sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x['sentiment'].sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not find sentiment")
)

# Step 5: Combine into a single unified chain
# RunnablePassthrough.assign keeps the original 'feedback' input and adds the 'sentiment' output
full_chain = (
    RunnablePassthrough.assign(sentiment=classifier_chain)
    | branch_chain
)

# Test the chain
user_input = {'feedback': 'This is a terrible smartphone with bad battery life.'}
print("Processing feedback...")
result = full_chain.invoke(user_input)

print("\n--- Final Response ---")
print(result)

full_chain.get_graph().print_ascii()