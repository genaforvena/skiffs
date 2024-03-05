import os
import subprocess
from typing import List, Tuple

from llama_cpp import Llama
from transformers import pipeline
from util.finneganniser import finnegannise

MAX_TOKENS = 7000
GEMMA_HOME: str = os.environ.get("GEMMA_HOME")
LLAMA_HOME: str = os.environ.get("LLAMA_HOME")


class Bridge:
    def summarize(
        self, text: str, style: str, history: List[str] = []
    ) -> Tuple[str, List[str]]:
        summary = self._ask("Summary of the following text " + style, text)
        updated_history = history + ["User: " + text, "System: " + summary]
        tokens = sum(len(entry.split()) for entry in updated_history)
        while tokens > MAX_TOKENS and len(updated_history) > 2:
            # Remove the oldest entries (2 at a time)
            updated_history = updated_history[2:]
            tokens = sum(len(entry.split()) for entry in updated_history)
        return summary, updated_history

    def hallucinate(self, text: str, style: str = "") -> str:
        hallucination = self._ask(
            finnegannise(text),
            " and then the abstract black whole laughing after you said " + style + ":",
        )
        return hallucination

    def _ask(self, command_for: str, text: str) -> str:
        raise NotImplementedError

    @staticmethod
    def create(model: str) -> "Bridge":
        if model.endswith("gguf"):
            return LlamaBridge(model)
        elif model == "gemma.cpp":
            return GemmaBridge()
        else:
            return PipepileBridge(model)


class GemmaBridge(Bridge):
    def _ask(self, command_for: str, text: str) -> str:
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


class LlamaBridge(Bridge):
    def __init__(self, model: str):
        self._model = model

    def _ask(self, command_for: str, text: str) -> str:
        llama = Llama(
            LLAMA_HOME + "/models/" + self._model,
            chat_format="llama-2",
            verbose=False,
        )
        response = ""
        response = llama.create_chat_completion(
            messages=[{"role": "user", "content": command_for + " " + text}]
        )["choices"][0]["message"]["content"]

        return response


class PipepileBridge(Bridge):
    def __init__(self, model):
        self._model = model

    def _ask(self, command_for: str, text: str) -> str:
        raise NotImplementedError
