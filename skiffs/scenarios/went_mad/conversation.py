# This scenario is completely wrong as it uses text-cont\inuators to generate a conversation
# The proper way to do this is to use a conversation model to generate a conversation
# I'm keeping it just out of perverse pleasure I get from reading hallucinating machines.
from util import log
from nodes import text_continuator
import models_to_consider
import random

BOT_NAME = "PeggyAG"
DEFAULT_BOT_DESCRIPTION = (
    "Imaginary conversation between a user and an ODB impersonator."
)

out_file_name = "conversation.txt"


def _create_conservation_history_and_init_prompt(
    bot_description=DEFAULT_BOT_DESCRIPTION, bot_name=BOT_NAME
):
    conversation = f"""
    {bot_description}
    User: "Hey, if you had to choose a superhero power to make your morning routine easier, what would it be?"
    {bot_name}: "Ha! I'd pick teleportation, no doubt. Imagine just zappin' from the bed to the shower, then bam! – in the kitchen for some cereal. Life’s a wild ride, gotta move like lightning, you know?"
    User: "If you could only eat one meal for the rest of your life but it had to be something really mundane, what would you pick?"
    {bot_name}: "Yo, that's a trip! But I'd keep it real simple – plain rice, man. It's like life, basic but deep. You start plain, then mix it up with your own flavor, your own style. Rice is the canvas, life's the paint, ya feel me?"
    """

    return conversation


def generate_conversation(
    models,
    init_prompt=_create_conservation_history_and_init_prompt(),
    conversation_rounds=1000,
    times_model_speak_in_a_round=1,
    max_history=10,
):
    with open(out_file_name, "a+") as out_file:
        # Set reply to initial prompt as reply is used to continue the conversation
        conversation_history = [init_prompt]
        for _ in range(conversation_rounds):
            model_to_speak = random.choice(models)
            for _ in range(times_model_speak_in_a_round):
                # Format the conversation history and the model's prompt
                formatted_input = (
                    "\n".join(conversation_history[-max_history:]) + f"\n{BOT_NAME}: "
                )
                # Trim the prompt to fit the model's max input length
                formatted_input = formatted_input[
                    -text_continuator.get_model_max_input_length(model_to_speak) :
                ]
                # Generate the model's response
                response = text_continuator.generate_continuation(
                    model_to_speak, formatted_input, logging=False
                )
                reply = response.replace(model_to_speak, BOT_NAME)
                if reply.count(BOT_NAME) > 2:
                    reply = reply.replace(BOT_NAME, "", 1)
                # Add the model's response to the conversation history
                conversation_history.append(reply)
                # Log the conversation
                log(
                    out_file,
                    "\n\n" + model_to_speak + " replied: " + conversation_history[-1],
                )


if __name__ == "__main__":
    generate_conversation(models_to_consider.text_continuators)
