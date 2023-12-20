from typing import Callable, Dict, List
from numpy import pad
from torch import ne
from transformers import Conversation, AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
from models import models_to_consider
from util.text_utils import read_random_line
import random
import os
import re


def enough_words_in_reply(reply: str) -> bool:
    return len(reply.split(" ")) >= 4


default_conversation_starter = [
    {"role": "user", "content": "Do clouds ever feel lonely in the vast sky?"},
    {
        "role": "assistant",
        "content": "Perhaps they do, drifting along like solitary ships on an endless ocean of blue.",
    },
    {"role": "user", "content": "Is the moon a secret keeper of the night's whispers?"},
    {
        "role": "assistant",
        "content": "Indeed, it listens to the hushed tales of stars and the silent songs of the nocturnal world.",
    },
    {
        "role": "user",
        "content": "Can a melody paint the colors of a sunset in our minds?",
    },
    {
        "role": "assistant",
        "content": "Absolutely, each note a brushstroke of hues, crafting a visual symphony in our imagination.",
    },
    {"role": "user", "content": "Are shadows the earth's way of holding memories?"},
    {
        "role": "assistant",
        "content": "In a way, yes. Each shadow is a fleeting imprint, a gentle reminder of moments past.",
    },
    {
        "role": "user",
        "content": "Do trees whisper secrets to each other through their roots?",
    },
    {
        "role": "assistant",
        "content": "They do, sharing tales and wisdom in a language older than words, in the rustling language of leaves and earth.",
    },
    {
        "role": "user",
        "content": "Is the wind a messenger carrying stories from around the world?",
    },
    {
        "role": "assistant",
        "content": "It is indeed, a tireless traveler telling tales of distant lands, carrying whispers and scents from afar.",
    },
]

current_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
log_file = current_directory + "results/logs/conversation.log"

start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
default_max_conversation_length = 50000


def conversation_file() -> str:
    return current_directory + "results/conversations/" f"conversation_{start_time}.txt"


# TODO: Find a way to dynamically determine the max token limit from tokenizer config
max_model_token_limit = 1024


def _create_conversation_string(conversation_history: list[dict[str, str]]) -> str:
    return " ".join([msg["content"] for msg in conversation_history])


class Persona:
    def __init__(
        self,
        model_name: str,
        name: str,
        instructions: str = "",
        max_token_limit: int = max_model_token_limit,
    ) -> None:
        self.name = name
        self.model_name = model_name
        self.instructions = instructions
        self.tokenizer = None
        self.model = None
        self.max_token_limit = max_token_limit

    def generate_reply(self, conversation: Conversation) -> str:
        def _setup_generator() -> None:
            # God forgive me for this
            if "dialo" in self.model_name:
                padding = "right"  # DialoGPT performs better with padding on the right according to the docs
            else:
                padding = "left"
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, use_fast=False, padding_side=padding
            )
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        def _teardown_generator() -> None:
            self.tokenizer = None
            self.model = None

        _setup_generator()
        # Tokenize the conversation string
        inputs = self.tokenizer.encode(
            conversation.new_user_input,
            return_tensors="pt",
            max_length=max_model_token_limit,
            truncation=True,
        )

        # Determine a dynamic max_length for the output based on the input length
        input_length = inputs.size(1)
        output_max_length = min(
            input_length + 50, self.max_token_limit
        )  # Example formula
        # Generate response
        output_sequences = self.model.generate(
            input_ids=inputs,
            max_length=output_max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.95,
            top_k=60,
        )

        # Decode the response
        response = self.tokenizer.decode(
            output_sequences[:, inputs.shape[-1] :][0], skip_special_tokens=True
        )
        _teardown_generator()
        return response


def log(talker: Persona, reply: Dict[str, str]) -> None:
    log_message = (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        + talker.model_name
        + " "
        + str(reply)
    )
    with open(log_file, "a+") as f:
        f.write(log_message + "\n")
        print(log_message)
    conversation_entry = (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        + "\n"
        + talker.model_name
        + ": "
        + str(reply["content"])
    )
    with open(conversation_file(), "a+") as f:
        f.write(conversation_entry + "\n")


def _select_speaker(participants: list[Persona]) -> Persona:
    return random.choice(participants)


def _random_beckett_once(
    c: Conversation, talker: Persona, reply: str, _: list[Persona]
) -> str:
    if not enough_words_in_reply(reply):
        log(talker, {"role": "failed_assistant", "content": reply})
        return read_random_line(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "resources",
                "beckett_trilogy.txt",
            )
        )
    else:
        return reply


def _interagtor(
    conversation_obj: Conversation, talker: Persona, reply: str, _: list[Persona]
) -> str:
    while not enough_words_in_reply(reply):
        log(talker, {"role": "nothing_else_ever_assistant", "content": reply})
        reply = talker.generate_reply(conversation_obj)
    return reply


def _anyone_else(
    conversation_obj: Conversation, talker: Persona, reply: str, _: list[Persona]
) -> str:
    while not enough_words_in_reply(reply):
        log(talker, {"role": "fail_better_worse_again_assistant", "content": reply})
        random_talker = random.choice(participants)
        reply = random_talker.generate_reply(conversation_obj)
    return reply


def talk(
    participants: list[Persona],
    conversation_history: list[dict[str, str]],
    conversation_rounds: int = default_max_conversation_length,
    _select_speaker: Callable[[list[Persona]], Persona] = _select_speaker,
    _select_reply=_random_beckett_once,
) -> None:
    for i in range(conversation_rounds):
        conversation_obj = _create_conresation_obj(conversation_history)
        talker = _select_speaker(participants)
        reply = talker.generate_reply(conversation_obj)
        reply = _select_reply(conversation_obj, talker, reply, participants)
        # Determine the role based on the last message in the conversation object```
        role = "user" if i % 2 == 0 else "assistant"
        new_message = {"role": role, "content": reply}
        conversation_history.append(
            new_message
        )  # Append the new message to the history

        log(talker, new_message)

        # Update the conversation object with the new message
        if role == "user":
            conversation_obj.add_user_input(reply)
        else:
            conversation_obj.mark_processed()  # Mark the last assistant message as processed
            conversation_obj.append_response(reply)


def _create_conresation_obj(
    conversation: List[Dict[str, str]],
    max_length: int = default_max_conversation_length,
) -> Conversation:
    # Helper function to get the first sentence
    def get_first_sentence(text: str) -> str:
        # This will match the first sentence ending with a period, question mark, or exclamation point
        match = re.match(r"([^.!?]*[.!?])", text)
        return match.group(0) if match else text

    # Create a single string from the conversation
    conversation_str = " ".join(msg["content"] for msg in conversation)

    # Check if the conversation string exceeds the max length
    if len(conversation_str) > max_length:
        # Keep only the last two items
        conversation = conversation[-2:]
        # Retain only the first sentence of each of the last two items
        for i in range(len(conversation)):
            conversation[i]["content"] = get_first_sentence(conversation[i]["content"])

    # Convert the conversation history list to a Conversation object
    conversation_obj = Conversation()
    for msg in conversation:
        if msg["role"] == "user":
            conversation_obj.add_user_input(msg["content"])
        else:
            conversation_obj.mark_processed()  # Mark the last assistant message as processed
            conversation_obj.append_response(msg["content"])

    return conversation_obj


if __name__ == "__main__":
    participants = [
        Persona(model_name=model_name, name=model_name)
        for model_name in models_to_consider.conversation_models
    ]

    talk(
        participants=participants,
        conversation_history=default_conversation_starter,
        _select_reply=_anyone_else,
    )
