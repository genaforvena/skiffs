import os
import random
import json
import pathlib
import math
import nltk
import argparse
from transformers import pipeline
from datetime import datetime
from typing import List

from transformers import AutoTokenizer

from models import models_to_consider, picked_models
from util.text_utils import calculate_entropy

DEFAULT_SUMMARY_MIN_LENGTH = 1


class Checkpoint:
    def __init__(self, current_chunk_index=0, processed_chunks=[]):
        self.current_chunk_index = current_chunk_index
        self.processed_chunks = processed_chunks

    def save(self, file_path):
        with open(file_path, "w") as file:
            json.dump(self.__dict__, file)

    @staticmethod
    def load(file_path):
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
                return Checkpoint(**data)
        except FileNotFoundError:
            return Checkpoint()


class Summarizer:
    def __init__(
        self,
        summarization_model_names: list[str],
        keyword_extraction_model_names: list[str],
        hallucination_models: list[str] = [],
        convert_to_headline: bool = False,
        hallucination_times: int = 0,
        ask_persianmind: bool = False,
        russian: bool = False,
    ) -> None:
        self.summarization_model_name = summarization_model_names[0]
        self.keyword_extraction_model_name = keyword_extraction_model_names[0]
        self.hallucination_models = hallucination_models
        self.creation_time = datetime.now()
        self.convert_to_headline = convert_to_headline
        self.hallucination_times = hallucination_times
        self.ask_persianmind = ask_persianmind
        self.russian = russian
        nltk.download("punkt")

    def summarize(self, name: str, txt: str, min_length: int = 1) -> str:
        self.log_name = self._create_out_filename(name, "log")
        self.merged_summary_file_name = self._create_out_filename(
            name, format="md", postfix="result_using_merge"
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
        self._log("Summarization models summary: \n" + summary + "\n")
        if self.russian:
            keywords = ""
        else:
            keywords_extractor = pipeline(
                "summarization", model=self.keyword_extraction_model_name
            )
            keywords = keywords_extractor(text)[0]["summary_text"]
            self._log("Keywords extractor summary: \n" + keywords + "\n")

        if keywords == "":
            pass
        else:
            summary = summarizator(
                summary + "\n" + keywords,
                max_length=math.floor(summarizator_max_length / 12),
                do_sample=False,
            )[0]["summary_text"]
            self._log("Summary with keywords: \n" + summary + "\n")
        if self.hallucination_times > 0:
            times = self.hallucination_times
            while times > 0:
                model_to_hallucinate = random.choice(self.hallucination_models)
                old_summary = summary
                summary = pipeline(
                    "text-generation",
                    trust_remote_code=True,
                    model=model_to_hallucinate,
                )(summary, max_length=summarizator_max_length / (6 + times))[0][
                    "generated_text"
                ]
                entropy = calculate_entropy(summary.replace(old_summary, ""))
                if entropy < 0.5:
                    self._log(
                        "Rejected summary: \n"
                        + "by the model "
                        + model_to_hallucinate
                        + "\n"
                        + "because the entropy is "
                        + str(entropy)
                        + "\n"
                        + "Hallucinated summary: \n"
                        + "by the model "
                        + model_to_hallucinate
                        + "\n"
                        + summary
                        + " \n + after "
                        + str(times)
                        + " times \n"
                    )
                    continue
                self._log(
                    "Hallucinated summary: \n"
                    + "by the model "
                    + model_to_hallucinate
                    + "\n"
                    + summary
                    + " \n + after "
                    + str(times)
                    + " times \n"
                )
                times -= 1
            # Lets keep the original summary still
            # hallucinated_summary = hallucinated_summary.replace(summary, "")
        if self.convert_to_headline is True:
            headline_pipeline = pipeline(
                "text-generation",
                trust_remote_code=True,
                model=models_to_consider.story_tellers[0],
            )
            headline_max_length = math.floor(
                headline_pipeline.tokenizer.model_max_length
            )
            # this 9 is quite arbitrary number to avoid overflow
            summary = headline_pipeline(summary, max_length=headline_max_length - 9)[0][
                "generated_text"
            ]
        if self.ask_persianmind:
            summary = ask_persianmind(summary)
        return summary

    def _merge_summarize(
        self, name: str, texts: List[str], summary_min_length: int
    ) -> str:
        checkpoint_file = "checkpoint.json"
        checkpoint = Checkpoint.load(checkpoint_file)

        iteration = 0
        while len(texts) > summary_min_length:
            merged_texts = []
            self._print_out(
                "\n\n\n\nIteration " + str(iteration) + "\n\n",
            )
            for i in range(checkpoint.current_chunk_index, len(texts), 2):
                combined_text = texts[i]
                if i + 1 < len(texts):
                    combined_text += "\n\n" + texts[i + 1]
                self._log("Merging and summarizing: \n" + combined_text)
                merged_summary = self._call_summarizer(combined_text)
                merged_texts.append(merged_summary)
                self._log("\n\nMerged Summary: \n" + merged_summary + "\n\n\n\n")
                self._print_out("\n" + merged_summary + "\n")

                checkpoint.current_chunk_index = i + 1
                checkpoint.processed_chunks.append(merged_summary)
                checkpoint.save(checkpoint_file)
            checkpoint.current_chunk_index = 0
            checkpoint.save(checkpoint_file)
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

    def _create_out_filename(
        self, source_name: str, format: str = "txt", postfix: str = ""
    ) -> str:
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
            + "."
            + format
        )
        return output_filename


def ask_persianmind(prompt: str) -> str:
    from transformers import LlamaTokenizer, LlamaForCausalLM
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LlamaForCausalLM.from_pretrained(
        "universitytehran/PersianMind-v1.0",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": device},
    )
    tokenizer = LlamaTokenizer.from_pretrained(
        "universitytehran/PersianMind-v1.0",
    )

    TEMPLATE = "{context}\nYou: {prompt}\nPersianMind: "
    CONTEXT = (
        "This is a conversation with PersianMind. It is an artificial intelligence model designed by a team of "
        "NLP experts at the University of Tehran to help you with various tasks such as answering questions, "
        "providing recommendations, and helping with decision making. You can ask it anything you want and "
        "it will do its best to give you accurate and relevant information."
    )

    model_input = TEMPLATE.format(context=CONTEXT, prompt=prompt)
    input_tokens = tokenizer(model_input, return_tensors="pt")
    input_tokens = input_tokens.to(device)
    generate_ids = model.generate(
        **input_tokens, max_new_tokens=512, do_sample=False, repetition_penalty=1.1
    )
    model_output = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return model_output[len(model_input) :].replace(prompt, "")


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
    args.add_argument(
        "--hallucination-times",
        type=int,
        help="How many times after completing the summary should we hallucinate, default 0",
        default=0,
    )
    args.add_argument(
        "--ask-persianmind",
        type=bool,
        help="Ask persianmind for help",
        default=False,
    )
    args.add_argument(
        "--convert-to-headline",
        type=str,
        help="Name of the model to write headline",
        default="",
    )
    args.add_argument(
        "--russian",
        type=bool,
        help="Use russian models",
        default=False,
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

    if args.parse_args().russian:
        models_for_summarization = ["IlyaGusev/mbart_ru_sum_gazeta"]
        keywords_extraction_model_name = []
        hallucination_models = [
            "Mary222/MADE_AI_Dungeon_model_RUS",
            "igorktech/rugpt3-joker-150k",
            "Nehc/gpt2_lovecraft_ru",
        ]
    else:
        hallucination_models = models_to_consider.hallucinators

    keywords_extraction_model_name = picked_models.keyword_extraction_models
    min_summary_length = args.parse_args().min_length
    print("Summarizing text from path: " + src)
    summary = open(
        src,
        "r",
    ).read()
    summarizator = Summarizer(
        models_for_summarization,
        keywords_extraction_model_name,
        hallucination_models=hallucination_models,
        hallucination_times=args.parse_args().hallucination_times,
        convert_to_headline=args.parse_args().convert_to_headline,
        ask_persianmind=args.parse_args().ask_persianmind,
        russian=args.parse_args().russian,
    )
    summary = summarizator.summarize(
        src.split("/")[-1].split(".")[0].lower(), summary, min_summary_length
    )
