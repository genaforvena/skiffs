import subprocess
import os
from typing import List, Tuple

MAX_TOKENS = 7000
GEMMA_HOME = os.environ.get("GEMMA_HOME")


def summarize(
    text: str, style: str = "", history: List[str] = []
) -> Tuple[str, List[str]]:
    summary = _ask_gemma("Summary of the following text " + style, text)
    updated_history = history + ["User: " + text, "System: " + summary]
    tokens = sum(len(entry.split()) for entry in updated_history)
    while tokens > MAX_TOKENS and len(updated_history) > 2:
        # Remove the oldest entries (2 at a time)
        updated_history = updated_history[2:]
        tokens = sum(len(entry.split()) for entry in updated_history)
    return summary, updated_history


def hallucinate(text: str, style: str = "") -> str:
    hallucination = _ask_gemma(
        text,
        " and then the abstract black whole laughing after you said " + style + ":",
    )
    return hallucination


def _ask_gemma(command_for: str, text: str) -> str:
    if GEMMA_HOME is None:
        raise ValueError("GEMMA_HOME not set")
    command = [
        "echo",
        "'" + command_for + ": " + text + "'",
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
    return response
