import os
import subprocess
from typing import List, Tuple
from nltk import pr

from transformers import pipeline
from util.finneganniser import finnegannise

MAX_TOKENS = 7000
GEMMA_HOME: str = os.environ.get("GEMMA_HOME")
LLAMA_HOME: str = os.environ.get("LLAMA_HOME")


class Bridge:
    def summarize(
        self, text: str, style: str, history: List[str] = []
    ) -> Tuple[str, List[str]]:
        try:
            prompt = "Summarize the following text " + style + ":"
            summary = self._ask(prompt, text, 300)
            if text in summary:
                summary = summary.replace(text, "")
            if prompt in summary:
                summary = summary.replace(prompt, "")
            summary = summary + "\n\n"
        except Exception as e:
            print("Could not summarize", text)
            print(e)
            return text, history
        updated_history = history + ["User: " + text, "System: " + summary]
        tokens = sum(len(entry.split()) for entry in updated_history)
        while tokens > MAX_TOKENS and len(updated_history) > 2:
            # Remove the oldest entries (2 at a time)
            updated_history = updated_history[2:]
            tokens = sum(len(entry.split()) for entry in updated_history)
        return summary, updated_history

    def hallucinate(self, text: str, style: str) -> str:
        finnagannised_text = finnegannise(text)
        hallucination = self._ask(
            finnagannised_text,
            style + ":",
            30,
        )
        if finnagannised_text in hallucination:
            hallucination = hallucination.replace(finnagannised_text, "")
        return hallucination

    def _ask(self, command_for: str, text: str, max_new_tokens: int) -> str:
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
    def _ask(self, command_for: str, text: str, max_new_tokens: int) -> str:
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

    def _ask(self, command_for: str, text: str, max_new_tokens: int) -> str:
        from llama_cpp import Llama

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

    def _ask(self, command_for: str, text: str, max_new_tokens: int) -> str:
        pipe = pipeline(
            "text-generation",
            model=self._model,
        )
        tokenizer_kwargs = {
            "max_new_tokens": max_new_tokens,
            "truncation": False,
            "do_sample": True,
            "temperature": 0.3,
            "return_full_text": True,
        }
        try:
            if "phi" in self._model:
                prefix = "Exercise: "
                postfix = '"\nAnswer:'
            else:
                prefix = ""
                postfix = ""
            response = pipe(
                prefix + command_for + '\n\n"' + text + postfix, **tokenizer_kwargs
            )[0]["generated_text"]
        except Exception:
            return ""
        return response
