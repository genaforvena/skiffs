import models_to_consider
from datetime import datetime
from transformers import pipeline
from util import log

DEFAULT_MODEL_NAME = "gpt2"


def get_model_max_input_length(model_name):
    return pipeline("text-generation", model=model_name).model.config.max_length


def get_model_min_input_length(model_name):
    return pipeline("text-generation", model=model_name).model.config.min_length


def _get_last_reply(out):
    # Method stub for now
    return out.split("\n")[-1]


def generate_reply(
    model_name,
    message_to_reply_to,
    min_length=get_model_min_input_length(DEFAULT_MODEL_NAME),
    max_new_tokens=get_model_max_input_length(DEFAULT_MODEL_NAME),
    logging=True,
):
    generator = pipeline("text-generation", model=model_name)
    out = generator(
        message_to_reply_to,
        do_sample=True,
        min_length=min_length,
        max_new_tokens=max_new_tokens,
    )
    out = out[0]["generated_text"]

    # TODO get the last reply from the chat and use it has the prompt
    reply = _get_last_reply(out)
    return reply


if __name__ == "__main__":
    pass
