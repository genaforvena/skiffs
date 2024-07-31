import os
import random
import json
import pathlib
import argparse
from datetime import datetime
from typing import List, Tuple

from transformers import AutoTokenizer

from models import models_to_consider
from nodes import bridge

# from util.voice import narrate

DEFAULT_SUMMARY_MIN_LENGTH = 1


class Checkpoint:
    def __init__(
        self,
        result_file_name: str,
        checkpoint_file_path: str,
        current_chunk_index=0,
        processed_chunks=[],
    ):
        self.current_chunk_index = current_chunk_index
        self.processed_chunks = processed_chunks
        self.checkpoint_file_path = checkpoint_file_path
        self.result_file_name = result_file_name

    def save(self, file_path):
        with open(file_path, "w") as file:
            json.dump(self.__dict__, file)

    @staticmethod
    def load(result_file_name, checkpoint_file_path):
        try:
            with open(checkpoint_file_path, "r") as file:
                data = json.load(file)
                return Checkpoint(**data)
        except FileNotFoundError:
            return Checkpoint(
                result_file_name=result_file_name,
                checkpoint_file_path=checkpoint_file_path,
            )


class Summarizer:
    def __init__(
        self,
        summarizer_model_names: list[str],
        summary_style: str,
        summarization_rounds_per_chunk: int,
        hallucinator_model_names: list[str],
        hallucination_style: str,
        hallucination_rounds_per_chunk: int,
        narration_on: bool = False,
    ) -> None:
        self._summarizer_model_names = summarizer_model_names
        self._summary_style = summary_style
        self._summarization_rounds_per_chunk = summarization_rounds_per_chunk
        self._summary_memories = []
        self._hallucinator_model_names = hallucinator_model_names
        self._hallucination_style = hallucination_style
        self._hallucination_rounds_per_chunk = hallucination_rounds_per_chunk
        self._narration_on = narration_on
        self._creation_time = datetime.now()
        self._chunk_tokens_length = 0

    def summarize(self, src_file_path: str, txt: str, min_length: int = 1) -> str:
        self._log_file_name = self._create_out_filename(src_file_path, "log")
        self._result_file_name = self._create_out_filename(
            src_file_path, format="md", postfix="result_using_merge"
        )
        chunks = self._divide_text(txt)
        summary = self._merge_summarize(chunks, min_length)
        return summary

    def _restore_or_create_checkpoint(self) -> Tuple[Checkpoint, bool]:
        checkpoint_file = "checkpoint.json"
        checkpoint = Checkpoint.load(self._result_file_name, checkpoint_file)
        is_new_checkpoint = True
        if checkpoint.result_file_name is not None:
            is_new_checkpoint = False
            self._result_file_name = checkpoint.result_file_name
        return checkpoint, is_new_checkpoint

    def _merge_summarize(self, text_chunks: List[str], summary_min_length: int) -> str:
        checkpoint, is_new_checkpoint = self._restore_or_create_checkpoint()
        checkpoint_file = checkpoint.checkpoint_file_path
        iteration = 0
        while len(text_chunks) > summary_min_length:
            chunk_summaries = []
            if is_new_checkpoint:
                _write_to_result(
                    self._log_file_name,
                    self._result_file_name,
                    "\n\n\n\nIteration "
                    + str(iteration)
                    + "\n\n"
                    + "summarizer rounds: "
                    + str(self._summarization_rounds_per_chunk)
                    + "\n\n",
                )
            for i in range(checkpoint.current_chunk_index, len(text_chunks)):
                chunk_summary = text_chunks[i]
                chunk_summary = self._summarize_chunk(chunk_summary)
                chunk_summary = self._add_chunk_hallucinations_if_needed(chunk_summary)
                chunk_summaries.append(chunk_summary)
                _write_to_result(
                    self._log_file_name,
                    self._result_file_name,
                    chunk_summary,
                )
                #                if self._narration_on is True:
                #                    narrate(chunk_summary)

                checkpoint.current_chunk_index = i + 1
                checkpoint.processed_chunks.append(chunk_summary)
                checkpoint.save(checkpoint_file)
            checkpoint.current_chunk_index = 0
            checkpoint.save(checkpoint_file)
            text_chunks = chunk_summaries
            iteration += 1
        return text_chunks[0]

    def _summarize_chunk(self, text: str) -> str:
        summary = text
        for round in range(self._summarization_rounds_per_chunk):
            summarizer_model_name = random.choice(self._summarizer_model_names)
            _write_to_log(
                self._log_file_name,
                "\n\n\n\nSummarizing current round: "
                + str(round)
                + " model: "
                + summarizer_model_name
                + "\n\n"
                + "Text: "
                + text,
            )
            summary = self._call_summarizer(summarizer_model_name, summary)
            _write_to_log(
                self._log_file_name,
                "\n\n\n----------------->Summary after round "
                + str(round)
                + " by model "
                + summarizer_model_name
                + ": \n\n"
                + summary,
            )

        return summary

    def _call_summarizer(self, model: str, text: str) -> str:
        llm_bridge = bridge.Bridge.create(model, self._chunk_tokens_length)
        summary, memories = llm_bridge.summarize(text, self._summary_style)
        self._summary_memories += memories
        return summary

    def _add_chunk_hallucinations_if_needed(self, text: str) -> str:
        result = text
        for i in range(self._hallucination_rounds_per_chunk):
            hallucinator = random.choice(self._hallucinator_model_names)
            _write_to_log(
                self._log_file_name,
                "Hallucinating current round: "
                + str(i)
                + " model: "
                + hallucinator
                + "\n\n"
                + "Text: "
                + text,
            )
            hallucinated_continuation = self._call_hallucinator(
                hallucinator,
                result,
            )
            _write_to_log(
                self._log_file_name,
                "\n\n\n----------------->Hallucination after round "
                + str(i)
                + " by model "
                + hallucinator
                + ": \n\n"
                + hallucinated_continuation,
            )
            result = result + " " + hallucinated_continuation
            if "max_tokens" in hallucinated_continuation:
                break
        return result

    def _call_hallucinator(self, model: str, text: str) -> str:
        llm_bridge = bridge.Bridge.create(model, self._chunk_tokens_length)
        hallucination = llm_bridge.hallucinate(text, self._hallucination_style)
        return hallucination

    def _divide_text(self, text: str) -> List[str]:
        if self._summarizer_model_names[0].endswith(".gguf"):
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        elif "OpenELM" in self._summarizer_model_names[0]:
            # 'meta-llama/Llama-2-7b-hf' does not work on my machine
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        elif "Octopus" in self._summarizer_model_names[0]:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        elif "Oute" in self._summarizer_model_names[0]:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        elif "Smol" in self._summarizer_model_names[0]:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self._summarizer_model_names[0], trust_remote_code=True
            )
        # To make sure that style and command fits into the model
        self._chunk_tokens_length = tokenizer.model_max_length / 4

        max_token_length = self._chunk_tokens_length
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk_tokens = []

        for paragraph in paragraphs:
            sentence_tokens = tokenizer.tokenize(paragraph)

            if len(current_chunk_tokens) + len(sentence_tokens) <= max_token_length:
                current_chunk_tokens.extend(sentence_tokens)
            else:
                if current_chunk_tokens:
                    chunks.append(
                        tokenizer.convert_tokens_to_string(current_chunk_tokens)
                    )
                    current_chunk_tokens = sentence_tokens
                else:
                    chunks.append(
                        tokenizer.convert_tokens_to_string(
                            sentence_tokens[:max_token_length]
                        )
                    )
                    current_chunk_tokens = sentence_tokens[max_token_length:]

        if current_chunk_tokens:
            chunks.append(tokenizer.convert_tokens_to_string(current_chunk_tokens))

        return chunks

    def _create_out_filename(
        self, source_name: str, format: str = "txt", postfix: str = ""
    ) -> str:
        output_filename = (
            str(self._creation_time)
            + "summary_of_"
            + source_name.split("/")[-1]
            + "_"
            + postfix
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


def _write_to_log(file_name, msg: str) -> None:
    with open(file_name, "a+") as f:
        f.write(msg)
    print(msg)


def _write_to_result(log_file_name: str, result_file_name: str, msg: str) -> None:
    _write_to_log(log_file_name, msg)
    with open(result_file_name, "a+") as f:
        f.write(msg)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "src_file_path",
        type=str,
        help="Source file",
    )
    args.add_argument(
        "--summarization-rounds-per-chunk",
        type=int,
        help="Number of rounds of summarizer per chunk, default 1",
        default=1,
    )
    args.add_argument(
        "--hallucination-rounds-per-chunk",
        type=int,
        help="Number of times hallucinator models, if passed will continue each summarized chunk, default 0",
        default=0,
    )
    args.add_argument(
        "--summarizer-models",
        type=list,
        help="List of models to use for summarization. If empty gemma.cpp is used",
        default=[],
    )
    args.add_argument(
        "--hallucinator-models",
        type=list,
        help="List of models to use for continuation of the summary. If empty gemma.cpp is used",
        default=[],
    )
    args.add_argument(
        "--summary-style",
        type=str,
        help="Style of summary",
        default="in style of the source text",
    )
    args.add_argument(
        "--hallucination-style",
        type=str,
        help="Style of hallucinated continuation of the text",
        default="",
    )
    args.add_argument(
        "--narration-on",
        type=bool,
        help="Narrate the summary",
        default=False,
    )

    src_file_path = args.parse_args().src_file_path
    src_file_path = os.path.abspath(src_file_path)

    summarizer_models = args.parse_args().summarizer_models
    if summarizer_models == []:
        summarizer_models = models_to_consider.summarization_models
    hallucinator_models = args.parse_args().hallucinator_models
    if hallucinator_models == []:
        hallucinator_models = models_to_consider.hallucinators
    hallucination_rounds_per_chunk = args.parse_args().hallucination_rounds_per_chunk
    summary_style = args.parse_args().summary_style
    hallucination_style = args.parse_args().hallucination_style

    narration_on = args.parse_args().narration_on
    summarization_rounds_per_chunk = args.parse_args().summarization_rounds_per_chunk
    src_file = open(
        src_file_path,
        "r",
    ).read()
    print(
        "\n\nSummarizing file: "
        + src_file_path
        + " using style: "
        + summary_style
        + " and hallucination style: "
        + hallucination_style
        + " with summarizer models: "
        + str(summarizer_models)
        + " and hallucinator models: "
        + str(hallucinator_models)
        + " with summarization rounds per chunk: "
        + str(summarization_rounds_per_chunk)
        + " and hallucination rounds per chunk: "
        + str(hallucination_rounds_per_chunk)
        + " and narration on: "
        + str(narration_on)
    )
    summarizator = Summarizer(
        summarizer_models,
        summary_style,
        summarization_rounds_per_chunk,
        hallucinator_models,
        hallucination_style,
        hallucination_rounds_per_chunk,
        narration_on,
    )
    summarizator.summarize(
        src_file_path.split("/")[-1].split(".")[0].lower(),
        src_file,
    )
