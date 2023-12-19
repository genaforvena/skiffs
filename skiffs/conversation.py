from typing import Callable, Dict, List
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
        conversation_str = " ".join([msg["content"] for msg in conversation_history])

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
        conversation_history = _trim_conversation_if_needed(
            conversation_history, talker.tokenizer
        )


def _trim_conversation_if_needed(
    conversation: List[Dict[str, str]], tokenizer: AutoTokenizer, max_length: int = 1000
) -> List[Dict[str, str]]:
    # Ensure the last message is from the user
    if conversation and conversation[-1]["role"] != "user":
        conversation.pop()

    # Create a single string from the conversation
    conversation_str = " ".join(msg["content"] for msg in conversation)
    # Tokenize the string and check its length
    input_ids = tokenizer.encode(conversation_str, add_special_tokens=True)

    # If the length is too long, trim the conversation
    while len(input_ids) > max_length and len(conversation) > 1:
        # Remove messages from the start until we have 2 messages left
        conversation = conversation[-2:]
        conversation_str = " ".join(msg["content"] for msg in conversation)
        input_ids = tokenizer.encode(conversation_str, add_special_tokens=True)

        # If two messages are still too long, keep only the last user message
        if len(input_ids) > max_length and conversation[0]["role"] == "assistant":
            conversation = conversation[-1:]
            conversation_str = " ".join(msg["content"] for msg in conversation)
            input_ids = tokenizer.encode(conversation_str, add_special_tokens=True)

        # If one user message is too long, truncate it to fit the limit
        if len(input_ids) > max_length:
            # Truncate the tokens to the max_length, ensuring the last token is complete
            truncated_ids = input_ids[:max_length]
            # Decode tokens back to text
            truncated_text = tokenizer.decode(
                truncated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            conversation = [{"role": "user", "content": truncated_text}]
            break

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
