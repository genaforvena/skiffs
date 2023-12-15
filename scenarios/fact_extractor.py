from nltk import pr
import models_to_consider
from transformers import pipeline
from datetime import datetime
from text_utils import clean_and_cut, SUMMARIZE_CHUNK_SIZE


def extract_simple_facts(model_name, txt_path):
    print("Loading text from: " + txt_path + "\n")
    chunks = clean_and_cut(txt_path)
    
    summaries = []
    with open("resources/summaries.txt", "a") as f:
        for p in chunks:
            f.write("Model: " + model_name + "\n")
            f.write("Time: " + str(datetime.now()) + "\n")
            f.write("\n\n\n")
            print("\n\n\n")
            print("Chunk to sum up: " + p + "\n")

            summary = _run_summarization(p, model_name)
            summaries += summary
            print("Summary: " + summary + "\n")
            f.write("Chunk: " + p + "\n")
            f.write("Summary: " + summary + "\n")
            f.write("\n\n\n")


def _run_summarization(text, model):
    summarizer = pipeline("summarization", model=model)
    summary_text = summarizer(text, max_length=30, min_length=5)[0]["summary_text"]
    return summary_text


if __name__ == '__main__':
    for model in models_to_consider.summarization_models:
        extract_simple_facts(model, "resources/beckett_triology.txt")
