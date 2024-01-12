from datetime import datetime
import torch
import os
import random
import re
from typing import Callable, Dict, List, Tuple

from torch import instance_norm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Conversation,
    ReformerModelWithLMHead,
)

from models import models_to_consider, mamba
from util.text_utils import read_random_line


def enough_words_in_reply(reply: str) -> bool:
    return len(reply.split(" ")) >= 4


onlyfans_conversation_starter = [
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
    # Adding new entries based on the OnlyFans messaging guide
    {
        "role": "assistant",
        "content": "Welcome to my space! I'm thrilled to have you here. Let's embark on an exciting journey together!",
    },
    {
        "role": "assistant",
        "content": "Have you checked out my latest PPV content? It's something special that I've prepared just for you. Don't miss out!",
    },
    {
        "role": "assistant",
        "content": "What do you enjoy most about my content? I'm curious to know your thoughts!",
    },
    {
        "role": "assistant",
        "content": "Here's a special offer for you! Be the first to reply and get a custom video for free!",
    },
    {
        "role": "assistant",
        "content": "I've got exciting news! I'm now offering a new service on my OnlyFans. Stay tuned for more details!",
    },
]
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


def conversation_file() -> str:
    return current_directory + "results/conversations/" f"conversation_{start_time}.txt"


# TODO: Find a way to dynamically determine the max token limit from tokenizer config
MAX_TOKEN_LIMIT = 1024
MEMORY_RATIO = 0.2
DEFAULT_CONVERSATION_LENGTH = 5000


def _create_conversation_string(conversation_history: list[dict[str, str]]) -> str:
    return " ".join([msg["content"] for msg in conversation_history])


class Persona:
    def __init__(
        self,
        model_name: str,
        name: str,
        instructions: str = "",
        max_token_limit: int = MAX_TOKEN_LIMIT,
        first_memory="",
        memories: list[str] = [],
    ) -> None:
        self.name = name
        self.model_name = model_name
        self.instructions = instructions
        self.tokenizer = None
        self.model = None
        self.max_token_limit = max_token_limit
        self.first_memory = first_memory
        self.memories = memories

    def _setup_generator(self) -> None:
        # God forgive me for this
        if "dialo" in self.model_name:
            padding = "right"  # DialoGPT performs better with padding on the right according to the docs
        # God is dead, that is what Jesus said
        else:
            padding = "left"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, use_fast=False, padding_side=padding
        )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

    def _teardown_generator(self) -> None:
        self.tokenizer = None
        self.model = None

    def reply(self, conversation: Conversation) -> str:
        text_to_reply: str = (
            self._select_relevant_memory_snippet() + conversation.new_user_input
        )

        return self._generate_reply(text_to_reply)

    def _generate_reply(self, text: str) -> str:
        self._setup_generator()
        # Tokenize the conversation string
        inputs = self._encode(text)
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
        response = self._decode(output_sequences, inputs)

        self._teardown_generator()
        return response

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(
            text,
            return_tensors="pt",
            max_length=self.max_token_limit,
            truncation=True,
        )

    def _decode(self, output_sequences: list[int], inputs) -> str:
        return self.tokenizer.decode(
            output_sequences[:, inputs.shape[-1] :][0], skip_special_tokens=True
        )

    def _select_relevant_memory_snippet(self) -> str:
        if len(self.memories) > 0:
            return random.choice(self.memories)
        else:
            return self.first_memory

    def remember(self, memory: str) -> None:
        if len(self.first_memory) == 0:
            self.first_memory = memory
        self.memories.append(memory)
        if len(self.memories) > 10:
            self.memories.pop(0)


class ReformerPersona(Persona):
    def encode(self, list_of_strings: list[str], pad_token_id: int = 0):
        max_length = max([len(string) for string in list_of_strings])
        attention_masks = torch.zeros(
            (len(list_of_strings), max_length), dtype=torch.long
        )
        input_ids = torch.full(
            (len(list_of_strings), max_length), pad_token_id, dtype=torch.long
        )
        for idx, string in enumerate(list_of_strings):
            if not isinstance(string, bytes):
                string = str.encode(string)
            input_ids[idx, : len(string)] = torch.tensor([x + 2 for x in string])
            attention_masks[idx, : len(string)] = 1
        return input_ids, attention_masks

    def decode(self, outputs_ids):
        decoded_outputs = []
        for output_ids in outputs_ids.tolist():
            # transform id back to char IDs < 2 are simply transformed to ""
            decoded_outputs.append(
                "".join([chr(x - 2) if x > 1 else "" for x in output_ids])
            )
        return decoded_outputs[0]

    def reply(self, conversation: Conversation) -> str:
        model = ReformerModelWithLMHead.from_pretrained(self.model_name)
        encoded, attention_masks = self.encode([conversation.new_user_input])
        return self.decode(model.generate(encoded, do_sample=True, max_length=150))


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

    with open(conversation_file() + "only_replies", "a+") as f:
        f.write(reply["content"] + "\n")


def _select_speaker(participants: list[Persona]) -> Persona:
    return random.choice(participants)


def _random_beckett_once(
    c: Conversation, talker: Persona, reply: str, _: list[Persona]
) -> Tuple[str, Persona]:
    if not enough_words_in_reply(reply):
        log(talker, {"role": "failed_assistant", "content": reply})
        return (
            read_random_line(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "resources",
                    "beckett_trilogy.txt",
                )
            ),
            talker,
        )
    else:
        return reply, talker


def _interagtor(
    conversation_obj: Conversation, talker: Persona, reply: str, _: list[Persona]
) -> Tuple[str, Persona]:
    while not enough_words_in_reply(reply):
        log(talker, {"role": "nothing_else_ever_assistant", "content": reply})
        reply = talker.reply(conversation_obj)
    return reply, talker


def _anyone_else(
    conversation_obj: Conversation, talker: Persona, reply: str, _: list[Persona]
) -> Tuple[str, Persona]:
    random_talker = talker
    while not enough_words_in_reply(reply):
        log(talker, {"role": "fail_better_worse_again_assistant", "content": reply})
        random_talker = random.choice(participants)
        reply = random_talker.reply(conversation_obj)
    return reply, random_talker


def talk(
    participants: list[Persona],
    conversation_history: list[dict[str, str]],
    conversation_rounds: int = DEFAULT_CONVERSATION_LENGTH,
    _select_speaker: Callable[[list[Persona]], Persona] = _select_speaker,
    _select_reply=_random_beckett_once,
) -> None:
    for i in range(conversation_rounds):
        conversation_obj = _create_conresation_obj(conversation_history)
        talker = _select_speaker(participants)
        reply = talker.reply(conversation_obj)
        reply, talker = _select_reply(conversation_obj, talker, reply, participants)
        talker.remember(reply)
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
    max_length: int = DEFAULT_CONVERSATION_LENGTH,
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
        Persona(model_name, model_name, max_token_limit=1024)
        for model_name in [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ]
    ]
    for _ in range(DEFAULT_CONVERSATION_LENGTH):
        talk(
            participants=participants,
            conversation_history=onlyfans_conversation_starter,
            _select_reply=_anyone_else,
        )
