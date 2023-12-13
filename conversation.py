import text_continuator
import models_to_consider
import random

if __name__ == "__main__":
    conversation_rounds = 100
    prompt = "Say for be said. Missaid. From now say for missaid."
    each_model_rounds = 10
    for _ in range(conversation_rounds):
        text_continuator.generate_continuation(random.choice(models_to_consider.generative_models), prompt, conversation_rounds)
