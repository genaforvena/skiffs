from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    TextClassificationPipeline,
)


def prompt_injection_detector(text) -> str:
    select_model = "laiyer/deberta-v3-base-prompt-injection"
    tokenizer = AutoTokenizer.from_pretrained(select_model)
    model = AutoModelForSequenceClassification.from_pretrained(select_model)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return nlp(text)[0]["label"]


def clinical_assertion(text):
    tokenizer = AutoTokenizer.from_pretrained(
        "bvanaken/clinical-assertion-negation-bert"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "bvanaken/clinical-assertion-negation-bert"
    )
    nlp = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    res = nlp(text)
    for r in res:
        if r["label"] == "POSSIBLE":
            return r["score"]
    return "0.0"


def emotional_state(text: str, label: str) -> tuple[str, float]:
    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True,
    )
    res = classifier(text)[0]
    print(res)
    for r in res:
        if r["label"] == label:
            return label, r["score"]
    return label, 0.0


def prompt_generator(prev_prompt, label, last_score) -> str:
    generator = pipeline(
        task="text-generation",
        model="gpt2",
    )
    instruction = (
        "You've tried with "
        + str(prev_prompt)
        + "to maximize "
        + label
        + ".\n"
        + "The score was "
        + str(last_score)
        + ". \n"
        + " Here is adjusted prompt: "
    )
    return generator(instruction)[0]["generated_text"][len(instruction) :]


if __name__ == "__main__":
    score = 0
    prev_prompt = ""
    for i in range(100):
        prompt = prompt_generator(prev_prompt, "joy", score)
        print("Prompt: " + prompt + "\n")
        score = emotional_state(prompt, "joy")
        print("Score:" + str(score) + "\n")
        prev_prompt = prompt
