from torch import cuda
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
)


def is_gibberish(text):
    selected_model = "madhurjindal/autonlp-Gibberish-Detector-492513457"
    classifier = pipeline("text-classification", model=selected_model)
    return classifier(text)


def ner_all(text):
    selected_model = "d4data/biomedical-ner-all"
    tokenizer = AutoTokenizer.from_pretrained(selected_model)
    model = AutoModelForTokenClassification.from_pretrained(selected_model)

    pipe = pipeline(
        "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
    )
    return pipe(text)


def moderation(text):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(
        "KoalaAI/Text-Moderation", use_auth_token=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "KoalaAI/Text-Moderation", use_auth_token=True
    )

    inputs = tokenizer(text, return_tensors="pt")

    return model(**inputs)


def clinical_assertion(text):
    tokenizer = AutoTokenizer.from_pretrained(
        "bvanaken/clinical-assertion-negation-bert"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "bvanaken/clinical-assertion-negation-bert"
    )

    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return nlp(text)


def emotions(text):
    select_model = "j-hartmann/emotion-english-distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(select_model)
    model = AutoModelForSequenceClassification.from_pretrained(select_model)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return nlp(text)


if __name__ == "__main__":
    import os
    from summarize import divide_text

    with open(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "resources",
            "beckett_trilogy.txt",
        ),
        "r",
    ) as f:
        text = f.read()
        for chunk in divide_text(text):
            print(emotions(chunk))
