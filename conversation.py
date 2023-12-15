from huggingface_hub.repocard import model_index_to_eval_results
from util import log
from datetime import datetime
import text_continuator
import models_to_consider
import random

BOT_NAME = "PeggyAG"
out_file_name = "conversation.txt"
default_init_prompt = (
    "Let's start a conversation. Remember, you are a real human being with emotions, engage in a meaningful conversation.\n\nUser: What's the weather like today?\n"
    + BOT_NAME
    + ": It could be better I guess.\nUser: That's okay. Can you tell me a joke instead?\n"
    + BOT_NAME
    + ": Sure, here's one for you: Why don't scientists trust atoms? Because they make up everything!"
)


def generate_conversation(
    models,
    init_prompt=default_init_prompt,
    conversation_rounds=1000,
    times_model_speak_in_a_round=1,
    max_history=10,
):
    with open(out_file_name, "a+") as out_file:
        # Set reply to initial prompt as reply is used to continue the conversation
        conversation_history = [init_prompt]
        for _ in range(conversation_rounds):
            model_to_speak = random.choice(models)
            for _ in range(times_model_speak_in_a_round):
                # Format the conversation history and the model's prompt
                formatted_input = (
                    "\n".join(conversation_history[-max_history:]) + f"\n{BOT_NAME}: "
                )
                # Generate the model's response
                response = text_continuator.generate_continuation(
                    model_to_speak, formatted_input, logging=False
                )
                reply = response.replace(model_to_speak, BOT_NAME)
                # Add the model's response to the conversation history
                conversation_history.append(reply)
                # Log the conversation
                log(
                    out_file,
                    "\n\n" + model_to_speak + " replied: " + conversation_history[-1],
                )


if __name__ == "__main__":
    generate_conversation(models_to_consider.generative_models)
