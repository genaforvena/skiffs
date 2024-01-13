import os
import nltk
import argparse
import torch
from transformers import pipeline
from datetime import datetime
from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.configuration_utils import re

from models import models_to_consider, picked_models

DEFAULT_SUMMARY_MIN_LENGTH = 1


class Summarizer:
    def __init__(self, summarization_model_names: list[str]) -> None:
        self.summarization_model_names = summarization_model_names
        self.creation_time = datetime.now()
        self.outputs = []
        # Use only one summzriation model, ignore the provided list
        self.summarization_model_name = "Cohee/bart-factbook-summarization"

    def _sentence_tokenizer(self, text: str) -> List[str]:
        return nltk.tokenize.sent_tokenize(text)

    def _divide_text(self, text: str) -> List[str]:
        nltk.download("punkt")
        tokenizer = AutoTokenizer.from_pretrained(self.summarization_model_name)
        max_token_length = tokenizer.model_max_length
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk_tokens = []

        for paragraph in paragraphs:
            sentences = self._sentence_tokenizer(paragraph)
            for sentence in sentences:
                sentence_tokens = tokenizer.tokenize(sentence)

                # Check if adding this sentence exceeds the max token length
                if len(current_chunk_tokens) + len(sentence_tokens) <= max_token_length:
                    current_chunk_tokens.extend(sentence_tokens)
                else:
                    # Add the current chunk to chunks
                    if current_chunk_tokens:
                        chunks.append(
                            tokenizer.convert_tokens_to_string(current_chunk_tokens)
                        )
                        current_chunk_tokens = sentence_tokens
                    else:
                        # Handle very long sentences
                        chunks.append(
                            tokenizer.convert_tokens_to_string(
                                sentence_tokens[:max_token_length]
                            )
                        )
                        current_chunk_tokens = sentence_tokens[max_token_length:]

        # Add the last chunk if it exists
        if current_chunk_tokens:
            chunks.append(tokenizer.convert_tokens_to_string(current_chunk_tokens))

        return chunks

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
        for chunk in self._divide_text(txt):
            self._log("Summarizing: " + chunk)
            chunk_summary = self._call_summarizer(chunk)
            combined_summary += chunk_summary + "\n\n"

            self._log("\n\nSummary: \n" + chunk_summary + "\n\n\n\n")
            self._log(chunk_summary + "\n", summary_filename, False)

        return combined_summary

    def _call_summarizer(self, text: str, item_to_select_index: int = 0) -> str:
        summarizator = pipeline("summarization", model=self.summarization_model_name)

        summary = summarizator(text, max_length=256, do_sample=False)[0]["summary_text"]
        print("Rephrased summary: " + summary)
        keywords_extractor = pipeline(
            "summarization", model="transformer3/H1-keywordextractor"
        )
        keywords = keywords_extractor(text)[0]["summary_text"]
        print("Keywords: " + keywords)

        summary = summarizator(summary + keywords, max_length=128, do_sample=False)[0][
            "summary_text"
        ]
        return summary


class MergeSummarizer(Summarizer):
    def __init__(self, model_names: list[str]) -> None:
        super().__init__(model_names)
        self.simple_summarize = super().summarize

    def _merge_summarize(
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

    def summarize(self, name: str, txt: str, min_length: int = 1) -> str:
        chunks = self._divide_text(txt)
        initial_summaries = [self.simple_summarize(name, chunk) for chunk in chunks]
        final_summary = self._merge_summarize(name, initial_summaries, min_length)
        return final_summary


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
