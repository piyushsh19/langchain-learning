
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        local_files_only=True,
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    return tokenizer, generator


st.set_page_config(page_title="Local AI Research Tool")

st.title("Local AI Research Tool")
st.caption("Powered by TinyLlama — running locally")

try:
    tokenizer, generator = load_model()

    user_input = st.text_area("Enter your prompt")

    if st.button("Summarize", type="primary"):
        if not user_input.strip():
            st.warning("Please enter a prompt.")
        else:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful research assistant. "
                        "Answer clearly and concisely."
                    ),
                },
                {
                    "role": "user",
                    "content": user_input,
                },
            ]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            with st.spinner("Generating response locally..."):
                result = generator(
                    prompt,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.3,
                    return_full_text=False,
                )

            st.subheader("Response")
            st.write(result[0]["generated_text"].strip())

except Exception as error:
    st.error(f"Could not load or run the local model: {error}")
