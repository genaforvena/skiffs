import nltk
import re

SUMMARIZE_CHUNK_SIZE = 300


def get_last_sentence(text):
    sentences = get_sentences(text)
    # Iterate backwards through the sentences
    for sentence in reversed(sentences):
        # Check if the sentence meets the condition (e.g., has a certain length)
        if len(sentence) > 10:  # Adjust this condition as needed
            return sentence

    # If no sentence meets the condition, return None
    return "Somehow on. Till nohow on. Said nohow on."


def get_sentences(text):
    # Ensure NLTK sentence tokenizer is downloaded
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    if text is None:
        return "On. Somehow on. Till nohow on."
    # Tokenize the text into sentences
    sentences = nltk.tokenize.sent_tokenize(text)

    return sentences


def clean_and_cut(path, chunk_size=SUMMARIZE_CHUNK_SIZE):
    with open(path, "r") as f:
        txt = f.read()
        txt = re.sub(r'[A-Z]+', '', txt)
        txt = re.sub(r'\b\d+\b', '', txt)
        txt = re.sub(r'\[\d+\]', '', txt)  # Removes citation-like numbers e.g., [1], [2], etc.
        txt = re.sub(r'_{2,}', '', txt)    # Removes lines of underscores
        txt = re.sub(r'\*{2,}', '', txt)   # Removes lines of asterisks

        chunks = []
        while len(txt) > chunk_size:
            end = txt.rfind('.', 0, chunk_size)
            if end == -1:
                end = chunk_size
            else:
                end += 1
            chunks.append(txt[:end])
            txt = txt[end:].lstrip()
            if txt is not None:
                chunks.append(txt)

        return chunks

