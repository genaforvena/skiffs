import models_to_consider
from datetime import datetime
from transformers import pipeline
from nodes.personas.persona import Persona
from util import log

DEFAULT_MODEL_NAME = "gpt2"


def get_model_max_input_length(model_name):
    return pipeline("text-generation", model=model_name).model.config.max_length


def get_model_min_input_length(model_name):
    return pipeline("text-generation", model=model_name).model.config.min_length


def _get_last_reply(out):
    # Method stub for now
    return out.split("\n")[-1]


def _extract_last_message(message_histroy_string):
    return message_histroy_string.split("\n")[-1]


def _extract_conversation_context(message_histroy_string):
    return message_histroy_string.split("\n")[:-1]


def generate_reply(
    persona: Persona,
    message_histroy_string,
    min_length=get_model_min_input_length(DEFAULT_MODEL_NAME),
    max_new_tokens=get_model_max_input_length(DEFAULT_MODEL_NAME),
):
    last_reply = _get_last_reply(message_histroy_string)
    conversation_context = _extract_conversation_context(message_histroy_string)
    return persona.reply(conversation_context, last_reply)


if __name__ == "__main__":
    pass
