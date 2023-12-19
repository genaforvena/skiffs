from typing import Callable, Dict, List
from numpy import ma
from torch import selu
from transformers import Conversation, AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
from models import models_to_consider
import random

default_conversation_starter = [
    {"role": "user", "content": "Was john coltrane saint and beoynd good and evil?"},
    {
        "role": "assistant",
        "content": "you're so right! ole is the purest bliss there is!",
    },
]

output_file = "conversation.log"


def log(output_file: str, message: str) -> None:
    with open(output_file, "a") as f:
        f.write(message + "\n")
        print(message)


def _create_conversation_string(conversation_history: list[dict[str, str]]) -> str:
    return " ".join([msg["content"] for msg in conversation_history])


class Persona:
    def __init__(self, model_name: str, name: str, instructions: str = "") -> None:
        self.name = name
        self.model_name = model_name
        self.instructions = instructions
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_reply(self, conversation_history: List[Dict[str, str]]) -> str:
        # Join the conversation history into a single string
        conversation_str = _create_conversation_string(conversation_history)

        # Tokenize the conversation string
        inputs = self.tokenizer.encode(
            conversation_str, return_tensors="pt", max_length=1024, truncation=True
        )

        # Generate response
        output_sequences = self.model.generate(
            input_ids=inputs,
            max_length=1024,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.95,
            top_k=60,
        )

        # Decode the response
        response = self.tokenizer.decode(
            output_sequences[:, inputs.shape[-1] :][0], skip_special_tokens=True
        )
        return response


def _select_speaker(participants: list[Persona]) -> Persona:
    return random.choice(participants)


def talk(
    participants: list[Persona],
    conversation_history: list[dict[str, str]],
    conversation_rounds: int = 100,
    _select_speaker: Callable[[list[Persona]], Persona] = _select_speaker,
    output_file: str = output_file,
) -> None:
    # Convert the conversation history list to a Conversation object
    conversation_obj = Conversation()
    for msg in conversation_history:
        if msg["role"] == "user":
            conversation_obj.add_user_input(msg["content"])
        else:
            conversation_obj.mark_processed()  # Mark the last assistant message as processed
            conversation_obj.append_response(msg["content"])

    for i in range(conversation_rounds):
        talker = _select_speaker(participants)
        talker_message = talker.generate_reply(conversation_obj)
        # Determine the role based on the last message in the conversation object
        role = "user" if i % 2 == 0 else "assistant"
        new_message = {"role": role, "content": talker_message}
        conversation_history.append(
            new_message
        )  # Append the new message to the history

        # Log the message
        log_message = (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " " + str(new_message)
        )
        log(output_file, log_message)

        # Update the conversation object with the new message
        if role == "user":
            conversation_obj.add_user_input(talker_message)
        else:
            conversation_obj.mark_processed()  # Mark the last assistant message as processed
            conversation_obj.append_response(talker_message)

        # Trim the conversation if needed
        conversation_history = _trim_conversation_if_needed(conversation_history)


def _trim_conversation_if_needed(
    conversation: List[Dict[str, str]], max_length: int = 300
) -> List[Dict[str, str]]:
    conversation_str = _create_conversation_string(conversation)
    while len(conversation_str) > max_length:
        conversation.pop(0)
        conversation_str = _create_conversation_string(conversation)
        if len(conversation) < 2:
            break
    if len(conversation_str) > max_length:
        conversation = conversation[-2:]
        for msg in conversation:
            msg["content"] = msg["content"][: max_length // 2]
    return conversation


if __name__ == "__main__":
    participants = [
        Persona(model_name=model_name, name=model_name)
        for model_name in models_to_consider.conversation_models
    ]

    talk(
        participants=participants,
        conversation_history=default_conversation_starter,
    )
