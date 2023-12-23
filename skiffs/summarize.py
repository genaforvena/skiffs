from transformers import pipeline

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


def summarize(txt_path: str, model_name: str) -> str:
    summarizer = pipeline("summarization", model=model_name)
    txt = open(txt_path, "r").read()
    combined_summary = ""
    for chunk in divide_text(txt):
        combined_summary += (
            summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0][
                "summary_text"
            ]
            + "\n\n"
        )

    return combined_summary


if __name__ == "__main__":
    for model_name in models_to_consider.summarization_models:
        print(
            summarize(
                summarize(
                    summarize(
                        "skiffs/results/conversations/conversation_2023-12-23_05-38-45.txtonly_replies",
                        model_name,
                    ),
                    model_name,
                ),
                model_name,
            )
        )
