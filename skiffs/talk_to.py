from typing import List, Tuple
from llama_cpp import Llama


def ask(llama_path, text, history, max_tokens=1024) -> Tuple[str, List[str]]:
    llama = Llama(
        llama_path,
        chat_format="llama-2",
    )
    # Prepare the full text by combining history with the new text
    response = ""
    response = llama.create_chat_completion(
        messages=[{"role": "user", "content": text}]
    )["choices"][0]["message"]["content"]

    # Update the conversation history
    updated_history = history + ["User: " + text, "System: " + response]

    # Estimate the token count (simplified)
    tokens = sum(len(entry.split()) for entry in updated_history)

    # Trim the history if it exceeds the max token limit
    while tokens > max_tokens and len(updated_history) > 2:
        # Remove the oldest entries (2 at a time for user and Llama response)
        updated_history = updated_history[2:]
        tokens = sum(len(entry.split()) for entry in updated_history)

    return response, updated_history


if __name__ == "__main__":
    conversation_history = []

    llama_path = "/home/ilyahome/Developer/my_robots/llama.cpp/models/llava-v1.6-mistral-7b.Q5_K_M.gguf"
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting conversation.")
            break

        response, conversation_history = ask(
            llama_path, user_input, conversation_history
        )
        print("Llama:", response)
