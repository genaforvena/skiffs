from transformers import pipeline

from models import models_to_consider


def summarize(txt_path: str, model_name: str) -> str:
    summarizer = pipeline("summarization", model=model_name)
    txt = open(txt_path, "r").read()
    return summarizer(txt, max_length=130, min_length=30, do_sample=False)[0][
        "summary_text"
    ]


if __name__ == "__main__":
    for model_name in models_to_consider.summarization_models:
        print(
            summarize(
                "skiffs/results/conversations/conversation_2023-12-23_04-07-47.txtonly_replies",
                model_name,
            )
        )
