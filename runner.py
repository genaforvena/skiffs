import models_to_consider
from datetime import datetime
from transformers import pipeline

def _simple_next_prmpt(f, generator, prompt):
    f.write("Prompt: " + prompt + "\n")
    f.write("Generated: " + generator(prompt)[0]["generated_text"] + "\n")
    f.write("\n\n\n")
    return prompt


def run_model(model_name, init_prompt, rounds, prompt_get_func=_simple_next_prmpt):
    generator = pipeline("text-generation", model=model_name)
    prompt = init_prompt
    with open("out.txt", "a+") as f:
        f.write("Model: " + model_name + "\n")
        f.write("Time: " + str(datetime.now()) + "\n")
        f.write("\n\n\n")
        for _ in range(rounds):
            prompt = prompt_get_func(f, generator, prompt)
        f.write("\n\n\n")



if __name__ == "__main__":
    rounds = 100
    prompt = "Say for be said. Missaid. From now say for missaid."

    for model in models_to_consider.models:
        run_model(model, prompt, rounds)
