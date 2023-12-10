import re
from datetime import datetime, time
from transformers import pipeline


def split_into_sentences(text):
    sentences = re.split("(?<=[.!?]) +", text)
    return sentences


def run_model(model_name, prompt, rounds):
    generator = pipeline("text-generation", model=model_name)

    with open("out.txt", "a+") as f:
        f.write("\n\n\n")
        f.write("Model: " + model_name + "\n")
        f.write("Time: " + str(datetime.now()) + "\n")
        f.write("\n\n\n")
        for i in range(rounds):
            f.write("Prompt: " + prompt + "\n")
            out = generator(
                prompt,
                do_sample=True,
                min_length=20,
                max_new_tokens=100,
            )
            out = out[0]["generated_text"]
            # Delete prompt from output
            if prompt in out:
                print("Prompt: " + prompt)
                out = out[len(prompt) :]
            print(out)
            f.write(out + "\n")

            prompt = split_into_sentences(out)[-1]


if __name__ == "__main__":
    models = [
        "ToddGoldfarb/Cadet-Tiny",
        "KoboldAI/OPT-350M-Erebus",
        "PygmalionAI/pygmalion-350m",
        "bigscience/bloomz-560m",
        "cmarkea/bloomz-560m-sft-chat",
        "L-R/LLmRa-1.3B",
        "ericzzz/falcon-rw-1b-chat",
    ]
    rounds = 300
    prompt = "The most naughty thing I've ever done is"

    for model in models:
        run_model(model, prompt, rounds)
