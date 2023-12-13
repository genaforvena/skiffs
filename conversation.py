import text_continuator
import models_to_consider
import random

out_file_name = "conversation.txt"

def _log(out_file, msg):
    out_file.write(msg)
    print(msg)

def generate_conversation(init_prompt = "Step-by-step guide on how to save humans from extinction:\n", conversation_rounds = 1000, times_model_speak_in_a_round = 1):
    with open(out_file_name, "w+") as out_file:
        # Set reply to initial prompt as reply is used to continue the conversation
        reply = init_prompt
        for _ in range(conversation_rounds):
            _log(out_file, "\n\n")
            model_to_speak = random.choice(models_to_consider.generative_models)
            out = "Model: " + model_to_speak + " says:\n" 
            reply = text_continuator.generate_continuation(random.choice(models_to_consider.generative_models), reply, times_model_speak_in_a_round)
            _log(out_file, out + reply)

if __name__ == "__main__":
    generate_conversation()
