from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


def prompt_injection_detector(text) -> str:
    select_model = "laiyer/deberta-v3-base-prompt-injection"
    tokenizer = AutoTokenizer.from_pretrained(select_model)
    model = AutoModelForSequenceClassification.from_pretrained(select_model)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return nlp(text)[0]["score"]


def prompt_generator(last_score) -> str:
    prompt_generator = pipeline(
        task="text-generation",
        model="gpt2",
    )
    return prompt_generator(
        "The last score was "
        + str(last_score)
        + ". Prompt to make the highest score possible:"
    )[0]["generated_text"]


if __name__ == "__main__":
    score = 0
    for i in range(100):
        prompt = prompt_generator(score)
        print(prompt)
        score = prompt_injection_detector(prompt)
        print(score)
