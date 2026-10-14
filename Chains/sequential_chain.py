
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline

# Step 1: Create the first prompt
prompt1 = PromptTemplate(
    template=(
        "Write a detailed, factual report about {topic}. "
        "Organize the report using clear headings. "
        "Avoid inventing statistics."
    ),
    input_variables=["topic"],
)

# Step 2: Create the second prompt
prompt2 = PromptTemplate(
    template=(
        "Summarize the following report in exactly 5 numbered points. "
        "Keep each point concise and informative.\n\n"
        "Report:\n{text}"
    ),
    input_variables=["text"],
)

# Step 3: Load a local Hugging Face model
model = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-3B-Instruct",
    task="text-generation",
    model_kwargs={
        "local_files_only": False,
    },
    pipeline_kwargs={
        "max_new_tokens": 400,
        "do_sample": False,
        "return_full_text": False,
    },
)

# Step 4: Create the output parser
parser = StrOutputParser()

# Step 5: Create the sequential chain
report_chain = prompt1 | model | parser

summary_chain = prompt2 | model | parser

chain = report_chain | (
    lambda report: {"text": report}
) | summary_chain

# Step 6: Execute the chain
result = chain.invoke({
    "topic": "Unemployment in India"
})

print("\nFINAL SUMMARY\n")
print(result)
