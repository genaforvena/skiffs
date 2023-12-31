import re
from transformers import pipeline
from datetime import datetime
from typing import List

from models import models_to_consider


def divide_text(text: str, chunk_size: int = 256) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    last_word_index = 0

    for word in words:
        current_chunk.append(word)
        if word.endswith((".", "!", "?")):
            last_word_index = len(current_chunk) - 1

        if len(current_chunk) >= chunk_size:
            if len(current_chunk) == chunk_size and current_chunk[-1].endswith(
                (".", "!", "?")
            ):
                chunks.append(" ".join(current_chunk))
                current_chunk = []
            else:
                chunks.append(" ".join(current_chunk[:last_word_index]))
                current_chunk = current_chunk[last_word_index:]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def summarize(txt: str, model_name: str) -> str:
    summarizer = pipeline("summarization", model=model_name)
    combined_summary = ""
    for chunk in divide_text(txt):
        print("Summarizing: " + chunk)
        chunk_summary = summarizer(
            chunk, max_length=130, min_length=10, do_sample=False
        )[0]["summary_text"]
        combined_summary += chunk_summary + "\n\n"
        print("\n\n")
        print("Summary: " + chunk_summary)
        print("\n\n")
        print("\n\n")

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
