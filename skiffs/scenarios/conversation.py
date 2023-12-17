from typing import List, Callable
from nodes.personas.persona import Persona
import random
from util import log


output_file = "conversation.log"


def talk(
    participants: List[Persona] = [],
    conversation: List[str] = [],
    conversation_rounds: int = 100,
    talker_selector: Callable[[List[Persona]], Persona] = random.choice,
) -> None:
    for _ in range(conversation_rounds):
        talker = talker_selector(participants)
        new_message = talker.reply_to(conversation)
        log(output_file, new_message)
        conversation.append(new_message)
        conversation = _trim_conversation_if_needed(conversation)


def _trim_conversation_if_needed(conversation: List[str]) -> list[str]:
    if len(conversation) > 10:
        conversation.pop(0)
    return conversation


if __name__ == "__main__":
    import models_to_consider.conversation_models as conversation_models

    participants = [
        Persona(model_name=model_name, name=f"{model_name}{random.randint(1, 100)}")
        for model_name in conversation_models
    ]

    talk(participants=participants)
