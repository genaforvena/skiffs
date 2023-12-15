import models_to_consider
from datetime import datetime
from transformers import pipeline
from text_utils import get_last_sentence
from util import log


def get_model_max_input_length(model_name):
    return pipeline("text-generation", model=model_name).model.config.max_length


def generate_continuation(
    model_name, init_prompt, rounds=1, min_length=10, max_new_tokens=100, logging=True
):
    generator = pipeline("text-generation", model=model_name)
    prompt = init_prompt
    with open("out.txt", "a+") as f:
        if logging:
            log(f, "Model: " + model_name + "\n")
            log(f, "Time: " + str(datetime.now()) + "\n")
        for _ in range(rounds):
            if prompt is None:
                continue
            f.write("Prompt: " + prompt + "\n")
            out = generator(
                prompt,
                do_sample=True,
                min_length=min_length,
                max_new_tokens=max_new_tokens,
            )
            out = out[0]["generated_text"]
            if logging:
                log(f, "Generated: " + out)
            prompt = get_last_sentence(out)

        return prompt


if __name__ == "__main__":
    rounds = 100
    prompt = "Say for be said. Missaid. From now say for missaid."

    for model in models_to_consider.generative_models:
        generate_continuation(model, prompt, rounds)
