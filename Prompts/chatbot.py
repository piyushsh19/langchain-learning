
# from langchain_huggingface import (
#     ChatHuggingFace,
#     HuggingFacePipeline,
# )

# def load_model():
#     llm = HuggingFacePipeline.from_model_id(
#         model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#         task="text-generation",
#         pipeline_kwargs={
#             "max_new_tokens": 256,
#             "do_sample": True,
#             "temperature": 0.5,
#             "return_full_text": False,
#         },
#         model_kwargs={
#             "local_files_only": True,
#         },
#     )

#     return ChatHuggingFace(llm=llm)


# model = load_model()

# print("Local AI Chatbot (type 'exit' to quit)")
# chat_history = []
# while True:
#     user_input = input("You: ").strip()
#     chat_history.append(user_input)
#     if user_input.lower() == "exit":
#         print("Goodbye!")
#         break

#     if not user_input:
#         continue

#     try:
#         result = model.invoke(chat_history)
#         chat_history.append(result.content)
#         print("AI:", result.content)
#     except Exception as error:
#         print(f"Error: {error}")
# print(chat_history)
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFacePipeline,
)
from langchain_core.messages import HumanMessage, AIMessage


def load_model():
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 256,
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

print("Local AI Chatbot (type 'exit' to quit)")

chat_history = []

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    if not user_input:
        continue

    chat_history.append(HumanMessage(content=user_input))

    try:
        result = model.invoke(chat_history)

        chat_history.append(AIMessage(content=result.content))

        print("AI:", result.content)

    except Exception as error:
        chat_history.pop()  # Remove the failed user message
        print(f"Error: {error}")

print("\nChat History:")
for message in chat_history:
    print(f"{message.type}: {message.content}")
