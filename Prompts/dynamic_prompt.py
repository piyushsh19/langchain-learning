import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

st.set_page_config(page_title="Research Paper Summarizer")

# Load the local LLM only once
@st.cache_resource
def load_model():
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 500,
            "do_sample": True,
            "temperature": 0.5,
            "return_full_text": False,
        },
        model_kwargs={
            "local_files_only": True,
        },
    )

    return ChatHuggingFace(llm=llm)


model = load_model()

# Dynamic prompt template
prompt_template = PromptTemplate(
    input_variables=[
        "paper_input",
        "style_input",
        "length_input",
    ],
    template="""
Please summarize the research paper titled "{paper_input}".

Follow these specifications:

Explanation Style: {style_input}
Explanation Length: {length_input}

1. Mathematical Details:
- Include relevant mathematical equations if provided in the paper.
- Explain mathematical concepts using simple, intuitive code
  snippets where applicable.

2. Analogies:
- Use relatable analogies to simplify complex ideas.

3. Accuracy:
- If information is unavailable, state
  "Insufficient information available" instead of guessing.
- Ensure the summary is clear and accurate.

Important: If only a paper title is provided and the paper
content is unavailable, explain that you need the paper text
to summarize its actual findings. Do not invent paper contents.
"""
)

# Streamlit UI
st.title("Research Paper Summarizer")
st.caption("Powered by your locally cached TinyLlama model")

paper_input = st.text_input(
    "Research Paper Title",
    placeholder="e.g. Attention Is All You Need",
)

style_input = st.selectbox(
    "Explanation Style",
    [
        "Beginner-friendly",
        "Technical",
        "Mathematical",
        "Code-oriented",
        "Research-focused",
    ],
)

length_input = st.selectbox(
    "Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Detailed (section-by-section)",
    ],
)

# Generate the actual prompt
formatted_prompt = prompt_template.format(
    paper_input=paper_input,
    style_input=style_input,
    length_input=length_input,
)

# Preview the dynamic prompt
with st.expander("Preview Generated Prompt"):
    st.code(formatted_prompt, language="text")

if st.button("Generate Summary", type="primary"):
    if not paper_input.strip():
        st.warning("Please enter a research paper title.")
    else:
        with st.spinner("Generating summary locally..."):
            result = model.invoke(formatted_prompt)

        st.subheader("Generated Summary")
        st.write(result.content)
