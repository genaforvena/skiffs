from util import log
import text_continuator
import models_to_consider
import random

out_file_name = "conversation_gen_models.txt"


def generate_conversation(models, init_prompt = "The best story ever written:\n", conversation_rounds = 1000, times_model_speak_in_a_round = 1):
    with open(out_file_name, "a+") as out_file:
        # Set reply to initial prompt as reply is used to continue the conversation
        reply = init_prompt
        for _ in range(conversation_rounds):
            log(out_file, "\n\n")
            model_to_speak = random.choice(models)
            out = "Model: " + model_to_speak + " says:\n" 
            reply = text_continuator.generate_continuation(random.choice(models_to_consider.generative_models), reply, times_model_speak_in_a_round)
            log(out_file, out + reply)

if __name__ == "__main__":
    generate_conversation(models_to_consider.generative_models)
