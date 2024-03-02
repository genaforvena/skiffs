import subprocess
import os
from typing import List, Tuple

MAX_TOKENS = 7000
GEMMA_HOME = os.environ.get("GEMMA_HOME")


def summarize(text: str, history: List[str]) -> Tuple[str, List[str]]:
    response = ""
    # TODO: use take history into account
    if GEMMA_HOME is None:
        raise ValueError("GEMMA_HOME not set")
    command = [
        "echo",
        "'Summarize the following text. I need only summary text. Text: " + text + "'",
        "|",
        GEMMA_HOME
        + "/build/gemma -- --tokenizer "
        + GEMMA_HOME
        + "/build/tokenizer.spm --compressed_weights "
        + GEMMA_HOME
        + "/build/2b-it-sfp.sbs --model 2b-it --verbosity 0",
    ]
    response = subprocess.run(
        " ".join(command),
        capture_output=True,
        shell=True,
        text=True,
    ).stdout
    updated_history = history + ["User: " + text, "System: " + response]
    tokens = sum(len(entry.split()) for entry in updated_history)
    while tokens > MAX_TOKENS and len(updated_history) > 2:
        # Remove the oldest entries (2 at a time)
        updated_history = updated_history[2:]
        tokens = sum(len(entry.split()) for entry in updated_history)
    return response, updated_history
