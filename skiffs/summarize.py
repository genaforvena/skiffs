from transformers import pipeline
from datetime import datetime

from models import models_to_consider
import textwrap


def divide_text(text):
    # split the text into words
    words = text.split()
    # create a list to store the chunks
    chunks = []
    # set the chunk size
    chunk_size = 300
    # loop through the words
    for i in range(0, len(words), chunk_size):
        # create a chunk of 300 words
        chunk = words[i : i + chunk_size]
        # add the chunk to the list
        chunks.append(" ".join(chunk))
    # return the list of chunks
    return chunks


def summarize(txt: str, model_name: str) -> str:
    summarizer = pipeline("summarization", model=model_name)
    combined_summary = ""
    for chunk in divide_text(txt):
        chunk_summary = summarizer(
            chunk, max_length=130, min_length=10, do_sample=False
        )[0]["summary_text"]
        combined_summary += chunk_summary + "\n\n"
        print(chunk_summary)

    return combined_summary


if __name__ == "__main__":
    compression_times = 10
    for model_name in models_to_consider.summarization_models:
        print("Model:", model_name)
        print("Compressing")
        summary = open(
            "resources/beckett_trilogy.txt",
            "r",
        ).read()
        for i in range(compression_times):
            summary = summarize(summary, model_name)
            print("Summary after compression", i + 1, ":", summary)
            with open(
                "skiffs/results/summaries/beckett_trilogy_summary_"
                + str(model_name.split("/")[-1])
                + "_compression_"
                + str(i)
                + str(datetime.now())
                + ".txt",
                "w",
            ) as f:
                f.write(summary)
