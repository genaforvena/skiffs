import os
import pathlib
import math
import nltk
import argparse
from transformers import pipeline
from datetime import datetime
from typing import List

from transformers import AutoTokenizer

from models import models_to_consider, picked_models

DEFAULT_SUMMARY_MIN_LENGTH = 1


class Summarizer:
    def __init__(
        self,
        summarization_model_names: list[str],
        keyword_extraction_model_names: list[str],
    ) -> None:
        self.summarization_model_name = summarization_model_names[0]
        self.keyword_extraction_model_name = keyword_extraction_model_names[0]
        self.creation_time = datetime.now()
        nltk.download("punkt")

    def summarize(self, name: str, txt: str, min_length: int = 1) -> str:
        self.log_name = self._create_out_filename(name, "log")
        self.merged_summary_file_name = self._create_out_filename(
            name, "result_using_merge"
        )
        chunks = self._divide_text(txt)
        final_summary = self._merge_summarize(name, chunks, min_length)
        return final_summary

    def _call_summarizer(self, text: str) -> str:
        summarizator = pipeline("summarization", model=self.summarization_model_name)
        summarizator_max_length = math.floor(summarizator.tokenizer.model_max_length)

        summary = summarizator(
            text, max_length=math.floor(summarizator_max_length / 6)
        )[0]["summary_text"]
        keywords_extractor = pipeline(
            "summarization", model=self.keyword_extraction_model_name
        )
        keywords = keywords_extractor(text)[0]["summary_text"]

        summary = summarizator(
            summary + keywords,
            max_length=math.floor(summarizator_max_length / 12),
            do_sample=False,
        )[0]["summary_text"]
        return summary

    def _merge_summarize(
        self, name: str, texts: List[str], summary_min_length: int
    ) -> str:
        iteration = 0
        while len(texts) > summary_min_length:
            merged_texts = []
            self._print_out(
                "\n\n\n\nIteration " + str(iteration) + "\n\n",
            )
            for i in range(0, len(texts), 2):
                combined_text = texts[i]
                if i + 1 < len(texts):
                    combined_text += "\n\n" + texts[i + 1]
                self._log("Merging and summarizing: \n" + combined_text)
                merged_summary = self._call_summarizer(combined_text)
                merged_texts.append(merged_summary)
                self._log("\n\nMerged Summary: \n" + merged_summary + "\n\n\n\n")
                self._print_out("\n" + merged_summary + "\n")
            texts = merged_texts
            iteration += 1
        return texts[0]

    def _sentence_tokenizer(self, text: str) -> List[str]:
        return nltk.tokenize.sent_tokenize(text)

    def _divide_text(self, text: str) -> List[str]:
        tokenizer = AutoTokenizer.from_pretrained(self.summarization_model_name)
        max_token_length = tokenizer.model_max_length / 2.5
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

    def _log(self, msg: str) -> None:
        with open(self.log_name, "a+") as f:
            f.write(msg)
        print(msg)

    def _print_out(self, msg: str) -> None:
        self._log(msg)
        with open(self.merged_summary_file_name, "a+") as f:
            f.write(msg)

    def _create_out_filename(self, source_name: str, postfix: str = "") -> str:
        output_filename = (
            "summary_of_"
            + source_name.split("/")[-1]
            + "_"
            + postfix
            + "_at_"
            + str(self.creation_time)
        )
        output_filename = output_filename.replace(" ", "_").replace("/", "_")
        output_filename = (
            str(pathlib.Path(__file__).parent.parent.absolute())
            + "/tmp_results/summaries/"
            + output_filename
            + ".txt"
        )
        return output_filename


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "src",
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

    keywords_extraction_model_name = picked_models.keyword_extraction_models
    min_summary_length = args.parse_args().min_length
    print("Summarizing text from path: " + src)
    summary = open(
        src,
        "r",
    ).read()
    summarizator = Summarizer(models_for_summarization, keywords_extraction_model_name)
    summary = summarizator.summarize(
        src.split("/")[-1].split(".")[0].lower(), summary, min_summary_length
    )
