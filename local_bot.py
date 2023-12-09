import re
from transformers import pipeline

model_name = "bigscience/bloomz-560m"

generator = pipeline("text-generation", model=model_name)

prompt = "The most interesting thing I've ever done is"



def split_into_sentences(text):
    sentences = re.split("(?<=[.!?]) +", text)
    return sentences


with open("out.txt", "a+") as f:
    f.write(prompt + "\n" + model_name + "\n")
    for i in range(500):
        out = generator(
            prompt,
            do_sample=True,
            min_length=20,
            max_new_tokens=100,
        )
        out = out[0]["generated_text"]
        print(out)
        f.write(out + "\n")

        prompt = split_into_sentences(out)[-1]

