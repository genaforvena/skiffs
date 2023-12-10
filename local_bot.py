import re
import nltk
from datetime import datetime
from transformers import pipeline


def run_model(model_name, prompt, rounds):
    generator = pipeline("text-generation", model=model_name)
    with open("out.txt", "a+") as f:
        f.write("\n\n\n")
        f.write("Model: " + model_name + "\n")
        f.write("Time: " + str(datetime.now()) + "\n")
        f.write("\n\n\n")
        for _ in range(rounds):
            prompt = _do_run_and_get_next_prompt(f, generator, prompt)
        f.write("\n\n\n")


def _do_run_and_get_next_prompt(f, generator, prompt):
    f.write("Prompt: " + prompt + "\n")
    out = generator(
        prompt,
        do_sample=True,
        min_length=20,
        max_new_tokens=100,
    )
    out = out[0]["generated_text"]
    if out.startswith(prompt):
        print("Prompt: " + prompt)
        out = out[len(prompt) :]
    print(out)
    f.write(out + "\n")

    return _get_last_sentence(out)


def _get_last_sentence(text):
    # Ensure NLTK sentence tokenizer is downloaded
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    if text is None:
        return "On. Somehow on. Till nohow on."
    # Tokenize the text into sentences
    sentences = nltk.tokenize.sent_tokenize(text)

    # Iterate backwards through the sentences
    for sentence in reversed(sentences):
        # Check if the sentence meets the condition (e.g., has a certain length)
        if len(sentence) > 10:  # Adjust this condition as needed
            return sentence

    # If no sentence meets the condition, return None
    return "Somehow on. Till nohow on. Said nohow on."


if __name__ == "__main__":
    models = [
        # "KoboldAI/OPT-350M-Erebus",
        "PygmalionAI/pygmalion-350m",
        "bigscience/bloomz-560m",
        "cmarkea/bloomz-560m-sft-chat",
        "L-R/LLmRa-1.3B",
        "ericzzz/falcon-rw-1b-chat",
    ]
    rounds = 50
    prompt = "Say for be said. Missaid. From now say for missaid."

    for model in models:
        run_model(model, prompt, rounds)
