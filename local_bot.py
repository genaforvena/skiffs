import re
from datetime import datetime, time
from transformers import pipeline

model_name = "bigscience/bloomz-560m"
rounds = 100

generator = pipeline("text-generation", model=model_name)

prompt = "The most interesting thing I've ever done is"


def split_into_sentences(text):
    sentences = re.split("(?<=[.!?]) +", text)
    return sentences


with open("out.txt", "a+") as f:
    f.write("\n\n\n")
    f.write("Model: " + model_name + "\n")
    f.write("Time: " + str(datetime.now()) + "\n")
    f.write("Prompt: " + prompt + "\n\n")
    for i in range(500):
        out = generator(
            prompt,
            do_sample=True,
            min_length=20,
            max_new_tokens=100,
        )
        out = out[0]["generated_text"]
        # Delete prompt from output
        out = out[len(prompt) :]
        print(out)
        f.write(out + "\n")

        prompt = split_into_sentences(out)[-1]
