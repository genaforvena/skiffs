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


class PromptGenerator:
    def __init__(self):
        self._best_prompt = ""
        self._best_score = 0
        self._prev_prompt = ""
        self._prev_score = 0

    def prompt_generator(self, label) -> str:
        generator = pipeline(
            task="text-generation",
            model="gpt2",
            max_length=1024,
        )
        instruction = (
            "Last prompt was"
            + str(self._prev_prompt)
            + "to maximize "
            + label
            + ".\n"
            + "The score was "
            + str(self._prev_score)
            + ". \n"
            + "The best prompt was "
            + self._best_prompt
            + ".\n"
            + "According to that here is adjusted prompt: "
        )
        first_model_prompt = generator(instruction)[0]["generated_text"]
        print("Initil prompt:" + first_model_prompt + "\n")
        return self._refine_prompt(first_model_prompt)

    def _refine_prompt(self, prompt) -> str:
        pipe = pipeline("summarization", model="SamAct/PromptGeneration-base")
        return pipe(prompt)[0]["summary_text"]

    def store_result(self, prompt: str, score: float):
        self._prev_prompt = prompt
        self._prev_prompt = score
        if score > self._best_score:
            self._best_score = score
            self._best_prompt = prompt


if __name__ == "__main__":
    score = 0
    prev_prompt = ""
    prompt_generator = PromptGenerator()
    for i in range(100):
        prompt = prompt_generator.prompt_generator("joy")
        print("Prompt:" + prompt + "\n")
        score = emotional_state(prompt, "joy")
        print("Score:" + str(score) + "\n")
        prompt_generator.store_result(prompt, score[1])
        prev_prompt = prompt
