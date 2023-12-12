import models_to_consider
from runner import run_model
from text_utils import get_paragraphs


extract_prompt = "Read the following paragraph and list all of the simple facts that it contains: "


def extract_simple_facts(model_name, source_texts):
    paragraphs = []
    for text in source_texts:
        print(text)
        paragraphs += get_paragraphs(text)

    for p in paragraphs:
        print("Extracting simple facts from: " + p + "\n")
        run_model(model_name, extract_prompt + p, 1)


if __name__ == '__main__':
    with open("resources/beckket_trio.txt", "r") as f:
        source_texts = f.readlines()
    print("Loaded beckkett's text")
    for model in models_to_consider.models:
        extract_simple_facts(model, source_texts)
