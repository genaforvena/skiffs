import nltk

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

def get_paragraphs(text):
    import nltk.data
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    paragraphs = tokenizer.tokenize(text)
    return paragraphs

