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


def generate_reply(
    persona: Persona,
    conversation_history: str,
    min_length=get_model_min_input_length(DEFAULT_MODEL_NAME),
    max_new_tokens=get_model_max_input_length(DEFAULT_MODEL_NAME),
):
    return persona.add_reply_to(conversation_history, min_length, max_new_tokens)


if __name__ == "__main__":
    pass
