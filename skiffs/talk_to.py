from llama_cpp import Llama


def talk_to(llama, text, history, max_tokens=1024):
    # Prepare the full text by combining history with the new text
    full_text = "\n".join(history + ["User: " + text])
    response = llama(full_text)["choices"][0]["text"]

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
    llama = Llama(
        "/home/ilyahome/Developer/my_robots/llama.cpp/models/phi-2.Q4_K_M.gguf"
    )
    conversation_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting conversation.")
            break

        response, conversation_history = talk_to(
            llama, user_input, conversation_history
        )
        print("Llama:", response)
