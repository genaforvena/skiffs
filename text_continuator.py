import models_to_consider
from datetime import datetime
from transformers import pipeline
from text_utils import get_last_sentence


def generate_continuation(model_name, init_prompt, rounds):
    generator = pipeline("text-generation", model=model_name)
    prompt = init_prompt
    with open("out.txt", "a+") as f:
        f.write("Model: " + model_name + "\n")
        f.write("Time: " + str(datetime.now()) + "\n")
        f.write("\n\n\n")
        for _ in range(rounds):
            if prompt is None:
                continue
            f.write("Prompt: " + prompt + "\n")
            out = generator(
                prompt,
                do_sample=True,
                min_length=40,
                max_new_tokens=100,
            )
            out = out[0]["generated_text"]
            print("Generated: " + out)
            prompt = get_last_sentence(out)

        f.write("\n\n\n")



if __name__ == "__main__":
    rounds = 100
    prompt = "Say for be said. Missaid. From now say for missaid."

    for model in models_to_consider.generative_models:
        generate_continuation(model, prompt, rounds)
