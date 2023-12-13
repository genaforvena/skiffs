import text_continuator
import models_to_consider
import random

if __name__ == "__main__":
    conversation_rounds = 100
    last_models_reply = "Say for be said. Missaid. From now say for missaid."
    each_model_rounds = 5 
    for _ in range(conversation_rounds):
        model_to_speak = random.choice(models_to_consider.generative_models)
        print("Model: " + model_to_speak + " says:\n")
        last_models_reply = text_continuator.generate_continuation(random.choice(models_to_consider.generative_models), last_models_reply, each_model_rounds)
