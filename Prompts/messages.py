
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFacePipeline,
)
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)



def load_model():
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 128,
            "do_sample": False,
            "return_full_text": False,
        },
        model_kwargs={
            "local_files_only": True,
        },
    )

    return ChatHuggingFace(llm=llm)



def main():
    model = load_model()

    print("\nLocal AI Chatbot")
    print("Type 'exit' to quit.\n")

    messages = [
      SystemMessage(
    content=(
        "You are a helpful general-purpose assistant. "
        "Answer the user's latest question directly. "
        "Do not repeat previous answers unless asked. "
        "For arithmetic questions, calculate the result. "
        "Keep answers concise and relevant."
    )
)
    ]

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            if not user_input:
                continue

            messages.append(HumanMessage(content=user_input))

            try:
                result = model.invoke(messages)

                messages.append(AIMessage(content=result.content))

                print(f"\nAI: {result.content}\n")

            except Exception as error:
                messages.pop()
                print(f"Error generating response: {error}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
