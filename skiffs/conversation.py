from typing import Callable, Dict, List
from torch import ne
from transformers import Conversation, AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
from models import models_to_consider
from util.text_utils import read_random_line
import random
import os
import re


default_conversation_starter = [
    {"role": "user", "content": "Was john coltrane saint and beoynd good and evil?"},
    {
        "role": "assistant",
        "content": "you're so right! ole is the purest bliss there is!",
    },
]

current_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
log_file = current_directory + "results/logs/conversation.log"

start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
default_max_conversation_length = 50000


def conversation_file() -> str:
    return current_directory + "results/conversations/" f"conversation_{start_time}.txt"


max_reply_length = 150


def _create_conversation_string(conversation_history: list[dict[str, str]]) -> str:
    return " ".join([msg["content"] for msg in conversation_history])


class Persona:
    def __init__(self, model_name: str, name: str, instructions: str = "") -> None:
        self.name = name
        self.model_name = model_name
        self.instructions = instructions
        self.tokenizer = None
        self.model = None

    def generate_reply(self, conversation: Conversation) -> str:
        def _setup_generator() -> None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, use_fast=True, padding_side="left"
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
            max_length=max_reply_length,
            truncation=True,
        )

        # Generate response
        output_sequences = self.model.generate(
            input_ids=inputs,
            max_length=max_reply_length,
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


def talk(
    participants: list[Persona],
    conversation_history: list[dict[str, str]],
    conversation_rounds: int = default_max_conversation_length,
    _select_speaker: Callable[[list[Persona]], Persona] = _select_speaker,
) -> None:
    for i in range(conversation_rounds):
        conversation_obj = _create_conresation_obj(conversation_history)
        talker = _select_speaker(participants)
        talker_message = talker.generate_reply(conversation_obj)
        if talker_message == "":
            log(talker, {"role": "assistant", "content": "NO RESPONSE"})
            talker_message = read_random_line(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "resources",
                    "beckett_trilogy.txt",
                )
            )
        # Determine the role based on the last message in the conversation object```
        role = "user" if i % 2 == 0 else "assistant"
        new_message = {"role": role, "content": talker_message}
        conversation_history.append(
            new_message
        )  # Append the new message to the history

        log(talker, new_message)

        # Update the conversation object with the new message
        if role == "user":
            conversation_obj.add_user_input(talker_message)
        else:
            conversation_obj.mark_processed()  # Mark the last assistant message as processed
            conversation_obj.append_response(talker_message)


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
    )
