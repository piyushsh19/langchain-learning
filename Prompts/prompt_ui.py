
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFacePipeline,
)
import streamlit as st

st.set_page_config(page_title="Research Tool")

@st.cache_resource
def load_model():
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 1000,
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

st.header("Research Tool")

user_input = st.text_input("Enter the prompt")

if st.button("Search"):
    if user_input.strip():
        with st.spinner("Generating response..."):
            result = model.invoke(user_input)
        st.write(result.content)
    else:
        st.warning("Please enter a prompt.")
