import os
import argparse
from transformers import pipeline
from datetime import datetime
from typing import List

from models import models_to_consider, picked_models

DEFAULT_SUMMARY_MIN_LENGTH = 1


class Summarizer:
    def __init__(self, summarization_model_names: list[str]) -> None:
        self.summarization_model_names = summarization_model_names
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

    # At this point any pings of shame in me are gone and the code is ugly
    NOT_USED = 0

    def summarize(self, name: str, txt: str, min_length: int = NOT_USED) -> str:
        self._add_output(name, "detailed")
        summary_filename = self._add_output(name, "result")

        combined_summary = ""
        for chunk in divide_text(txt):
            self._log("Summarizing: " + chunk)
            chunk_summary = self._call_summarizer(chunk)
            combined_summary += chunk_summary + "\n\n"

            self._log("\n\nSummary: \n" + chunk_summary + "\n\n\n\n")
            self._log(chunk_summary + "\n", summary_filename, False)

        return combined_summary

    def _call_summarizer(self, text: str, item_to_select_index: int = 0) -> str:
        summaries = ""
        for summarization_model_name in self.summarization_model_names:
            summarizer = pipeline("summarization", model=summarization_model_name)
            summary = summarizer(text, max_length=36, min_length=4, do_sample=False)
            options_length = len(summary)

            for i in range(options_length):
                summaries += summary[i]["summary_text"]
        return summarizer(summaries, max_length=36, min_length=4, do_sample=False)[0][
            "summary_text"
        ]


class MergeSummarizer(Summarizer):
    def __init__(self, model_names: list[str]) -> None:
        super().__init__(model_names)
        self.simple_summarize = super().summarize

    def merge_summarize(
        self, name: str, texts: List[str], summary_min_length: int
    ) -> str:
        self.merged_summary_file_name = self._add_output(name, "merged")
        iteration = 0
        while len(texts) > summary_min_length:
            merged_texts = []
            self._log(
                "\n\n\n\nIteration " + str(iteration) + "\n\n",
                self.merged_summary_file_name,
            )
            for i in range(0, len(texts), 2):
                combined_text = texts[i]
                if i + 1 < len(texts):
                    combined_text += "\n\n" + texts[i + 1]
                self._log("Mrging and summarizing: \n" + combined_text)
                merged_summary = self._call_summarizer(combined_text)
                merged_texts.append(merged_summary)
                self._log("\n\nMerged Summary: \n" + merged_summary + "\n\n\n\n")
                self._log(
                    "\n" + merged_summary + "\n", self.merged_summary_file_name, False
                )
            texts = merged_texts
            iteration += 1
        return texts[0]

    def summarize(self, name: str, txt: str, min_length: int) -> str:
        chunks = divide_text(txt)
        initial_summaries = [self.simple_summarize(name, chunk) for chunk in chunks]
        final_summary = self.merge_summarize(name, initial_summaries, min_length)
        return final_summary


class KeywordAwareMergeSummarizer(MergeSummarizer):
    def __init__(self, model_names: list[str]) -> None:
        super().__init__(model_names)
        self.keyword_aware_summarize = super().summarize

    def summarize(self, name: str, txt: str, min_length: int) -> str:
        summarizer = pipeline("summarization", model=self.summarization_model_name)
        keywords = summarizer(txt, max_length=100, min_length=10, do_sample=False)[0][
            "summary_text"
        ]
        chunks = divide_text(txt)
        initial_summaries = [self.simple_summarize(name, chunk) for chunk in chunks]
        final_summary = self.merge_summarize(
            name, initial_summaries, min_summary_length
        )
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
    args = argparse.ArgumentParser()
    args.add_argument(
        "-src",
        type=str,
        help="Source file or directory to summarize. If directory, all files less than 100kb will be summarized.",
    )
    args.add_argument(
        "--code-summarization",
        type=bool,
        help="Summarize code instead of text",
        default=False,
    )
    args.add_argument(
        "--min-length",
        type=int,
        help="Minimum length of summary",
        default=DEFAULT_SUMMARY_MIN_LENGTH,
    )
    src = args.parse_args().src
    src = os.path.abspath(src)
    if os.path.isdir(src):
        # Read all files less than 100kb in the directory recoursevly and make a huge text out of them without any order
        print("Compressing directory " + src)
        summary = ""
        for root, dirs, files in os.walk(src):
            for file in files:
                if os.path.getsize(os.path.join(root, file)) < 100000:
                    summary += open(os.path.join(root, file), "r").read()

    if args.parse_args().code_summarization:
        models_for_summarization = models_to_consider.code_explanation_models
    else:
        models_for_summarization = picked_models.summarization_models

    min_summary_length = args.parse_args().min_length
    print("Compressing " + src)
    summary = open(
        src,
        "r",
    ).read()
    summarizator = MergeSummarizer(models_for_summarization)
    summary = summarizator.summarize(
        src.split("/")[-1].split(".")[0].lower(), summary, min_summary_length
    )
