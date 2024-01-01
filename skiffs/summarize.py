import os
from transformers import pipeline
from datetime import datetime
from typing import List

from models import models_to_consider


class Summarizer:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.creation_time = datetime.now()
        self.outputs = []

    def _log(
        self, msg: str, output_file_name: str = "", print_log: bool = True
    ) -> None:
        if output_file_name not in self.outputs:
            for out in self.outputs:
                if output_file_name in out:
                    output_file_name = out
                    break
        if output_file_name not in self.outputs:
            output_file_name = self._add_output("detailed")
        with open(output_file_name, "a+") as f:
            f.write(msg)
            if print_log:
                print(msg)

    def _add_output(self, source_name: str, postfix: str = "") -> str:
        output_file_name = (
            "summary_of_"
            + source_name.split("/")[-1]
            + "by_"
            + self.model_name
            + "_"
            + postfix
            + "_at_"
            + str(self.creation_time)
        )
        output_file_name = output_file_name.replace(" ", "_").replace("/", "_")
        output_file_name = (
            os.path.dirname(os.path.abspath(__file__))
            + "/results/summaries/"
            + output_file_name
            + ".txt"
        )
        self.outputs.append(output_file_name)
        return output_file_name

    def summarize(self, txt: str) -> str:
        summarizer = pipeline("summarization", model=self.model_name)
        self._add_output(self.model_name, "detailed")
        summary_filename = self._add_output(self.model_name, "result")

        combined_summary = ""
        for chunk in divide_text(txt):
            self._log("Summarizing: " + chunk)
            chunk_summary = summarizer(
                chunk, max_length=36, min_length=4, do_sample=False
            )[0]["summary_text"]
            combined_summary += chunk_summary + "\n\n"

            self._log("\n\nSummary: \n" + chunk_summary + "\n\n\n\n")
            self._log(chunk_summary + "\n", summary_filename, False)

        return combined_summary


class MergeSummarizer(Summarizer):
    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        self.simple_summarize = super().summarize

    def merge_summarize(self, texts: List[str]) -> str:
        summarizer = pipeline("summarization", model=self.model_name)
        while len(texts) > 1:
            merged_texts = []
            for i in range(0, len(texts), 2):
                combined_text = texts[i]
                if i + 1 < len(texts):
                    combined_text += "\n\n" + texts[i + 1]
                self._log("Merging and summarizing: \n" + combined_text)
                merged_summary = summarizer(
                    combined_text, max_length=100, min_length=10, do_sample=False
                )[0]["summary_text"]
                merged_texts.append(merged_summary)
                self._log("\n\nMerged Summary: \n" + merged_summary + "\n\n\n\n")
            texts = merged_texts
        return texts[0]

    def summarize(self, txt: str) -> str:
        chunks = divide_text(txt)
        initial_summaries = [self.simple_summarize(chunk) for chunk in chunks]
        final_summary = self.merge_summarize(initial_summaries)
        return final_summary


def divide_text(text: str, chunk_size: int = 256) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    last_word_index = 0

    for word in words:
        current_chunk.append(word)
        if word.endswith((".", "!", "?")):
            last_word_index = len(current_chunk) - 1

        if len(current_chunk) >= chunk_size:
            if len(current_chunk) == chunk_size and current_chunk[-1].endswith(
                (".", "!", "?")
            ):
                chunks.append(" ".join(current_chunk))
                current_chunk = []
            else:
                chunks.append(" ".join(current_chunk[:last_word_index]))
                current_chunk = current_chunk[last_word_index:]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


if __name__ == "__main__":
    compression_times = 1
    src = "resources/beckett_trilogy.txt"
    for model_name in models_to_consider.summarization_models:
        print("Model:", model_name)
        print("Compressing " + src)
        summary = open(
            src,
            "r",
        ).read()
        for i in range(compression_times):
            summarizator = MergeSummarizer(model_name)
            summary = summarizator.summarize(summary)
