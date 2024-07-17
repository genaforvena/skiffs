from collections import Counter
import math
import nltk
import re
import random
import os

SUMMARIZE_CHUNK_SIZE = 300

import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


def preprocess_text(text):
    words = word_tokenize(text.lower())

    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word.isalpha() and word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)


def cluster_sentences(sentences, n_clusters=5):
    preprocessed_sentences = [preprocess_text(sent) for sent in sentences]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(tfidf_matrix)

    clusters = [[] for _ in range(n_clusters)]
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(sentences[i])

    return clusters


def coherence_score(sentence1, sentence2):
    words1 = set(word_tokenize(sentence1.lower()))
    words2 = set(word_tokenize(sentence2.lower()))
    return len(words1.intersection(words2)) / len(words1.union(words2))


def organize_narrative(clusters):
    narrative = []
    for cluster in clusters:
        sorted_cluster = sorted(
            cluster,
            key=lambda x: sum(coherence_score(x, y) for y in cluster if x != y),
            reverse=True,
        )
        narrative.extend(sorted_cluster)
    return narrative


def clean_and_organize_text(input_file, output_file, n_clusters=5):
    with open(input_file, "r", encoding="utf-8") as file:
        text = file.read()

    sentences = sent_tokenize(text)

    valid_sentences = [sent for sent in sentences if len(word_tokenize(sent)) > 3]

    clusters = cluster_sentences(valid_sentences, n_clusters)

    organized_narrative = organize_narrative(clusters)

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(" ".join(organized_narrative))


def clean_text(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as file:
        text = file.read()

    text = re.sub(r"\s+", " ", text).strip()

    sentences = sent_tokenize(text)

    valid_sentences = [
        sent
        for sent in sentences
        if len(sent.split()) > 3 and any(c.isalpha() for c in sent)
    ]

    unique_sentences = list(dict.fromkeys(valid_sentences))

    cleaned_text = "\n".join(unique_sentences)
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(cleaned_text)


def calculate_entropy(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    word_counts = Counter(words)
    total_words = len(words)
    entropy = -sum(
        (count / total_words) * math.log2(count / total_words)
        for count in word_counts.values()
    )
    return entropy


def read_random_line(path):
    path = os.path.abspath(path)
    with open(path, "r") as f:
        lines = f.readlines()
        return random.choice(lines)


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
        txt = re.sub(r"[A-Z]+", "", txt)
        txt = re.sub(r"\b\d+\b", "", txt)
        txt = re.sub(
            r"\[\d+\]", "", txt
        )  # Removes citation-like numbers e.g., [1], [2], etc.
        txt = re.sub(r"_{2,}", "", txt)  # Removes lines of underscores
        txt = re.sub(r"\*{2,}", "", txt)  # Removes lines of asterisks

        chunks = []
        while len(txt) > chunk_size:
            end = txt.rfind(".", 0, chunk_size)
            if end == -1:
                end = chunk_size
            else:
                end += 1
            chunks.append(txt[:end])
            txt = txt[end:].lstrip()
            if txt is not None:
                chunks.append(txt)

        return chunks
