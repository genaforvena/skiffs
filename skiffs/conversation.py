from transformers import pipeline, Conversation
from typing import List, Callable
from datetime import datetime
from models import models_to_consider
import random


output_file = "conversation.log"


def log(file_name: str, message: str) -> None:
    with open(file_name, "a") as f:
        f.write(message + "\n")
        print(message)


class Persona:
    def __init__(
        self,
        model_name: str,
        name: str,
        instructions: str = "",
        past_memories: List[str] = [],
    ) -> None:
        self.name = name
        self.model_name = model_name
        self.instructions = instructions
        self.past_memories = past_memories
        self._pipeline = pipeline("conversational", model=model_name)

    def reply_to(self, conversation: List[str]) -> str:
        # Convert the conversation list to a Conversation object
        convo = Conversation("\n".join(conversation))
        out = self._pipeline(convo)
        # The output will be a Conversation object, you can get the new reply with:
        new_reply = out.generated_responses[-1]
        print(new_reply)
        return new_reply


def talk(
    participants: List[Persona] = [],
    conversation: List[str] = [],
    conversation_rounds: int = 100,
    _select_speaker: Callable[[List[Persona]], Persona] = random.choice,
) -> None:
    for _ in range(conversation_rounds):
        talker = _select_speaker(participants)
        new_message = talker.reply_to(conversation)
        log_message = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " " + new_message
        log(output_file, log_message)
        conversation.append(new_message)
        conversation = _trim_conversation_if_needed(conversation)


def _trim_conversation_if_needed(conversation: List[str]) -> list[str]:
    if len(conversation) > 10:
        conversation.pop(0)
    return conversation


if __name__ == "__main__":
    participants = [
        Persona(model_name=model_name, name=model_name)
        for model_name in models_to_consider.conversation_models
    ]

    talk(
        participants=participants,
        conversation=["Never told a truth in your life? Can't start now!"],
    )