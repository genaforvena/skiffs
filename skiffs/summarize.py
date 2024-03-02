import os
import random
import json
import pathlib
import nltk
import argparse
from datetime import datetime
from typing import List

from transformers import AutoTokenizer

from models import picked_models
from util.finneganniser import finnegannise
import nodes.gemma_bridge as gemma
import talk_to as llama

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

    def summarize(self, src_file_path: str, txt: str, min_length: int = 1) -> str:
        nltk.download("punkt")
        self._log_file_name = self._create_out_filename(src_file_path, "log")
        self._result_file_name = self._create_out_filename(
            src_file_path, format="md", postfix="result_using_merge"
        )
        chunks = self._divide_text(txt)
        summary = self._merge_summarize(chunks, min_length)
        return summary

    def _restore_or_create_checkpoint(self) -> Checkpoint:
        checkpoint_file = "checkpoint.json"
        checkpoint = Checkpoint.load(self._result_file_name, checkpoint_file)
        if checkpoint.result_file_name is not None:
            self._result_file_name = checkpoint.result_file_name
        return checkpoint

    def _merge_summarize(self, text_chunks: List[str], summary_min_length: int) -> str:
        checkpoint = self._restore_or_create_checkpoint()
        checkpoint_file = checkpoint.checkpoint_file_path
        iteration = 0
        while len(text_chunks) > summary_min_length:
            chunk_summaries = []
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
                current_chunk = text_chunks[i]
                chunk_summary = self._summarize_chunk(current_chunk)
                chunk_summary = self._hallucinate_chunk(
                    chunk_summary, self._hallucination_style
                )
                chunk_summaries.append(chunk_summary)
                _write_to_result(
                    self._log_file_name,
                    self._result_file_name,
                    "\n" + chunk_summary + "\n",
                )
                if self._narration_on is True:
                    from skiffs.util.voice import narrate

                    narrate(chunk_summary)

                checkpoint.current_chunk_index = i + 1
                checkpoint.processed_chunks.append(chunk_summary)
                checkpoint.save(checkpoint_file)
            checkpoint.current_chunk_index = 0
            checkpoint.save(checkpoint_file)
            text_chunks = chunk_summaries
            iteration += 1
        return text_chunks[0]

    def _summarize_chunk(self, text: str) -> str:
        summary = ""
        for round in range(self._summarization_rounds_per_chunk):
            if self._summarizer_model_names != []:
                summarizer_model_name = random.choice(self._summarizer_model_names)
            else:  # If the model is not set, use gemma.cpp
                summarizer_model_name = "gemma.cpp"
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
            summary = self._call_summarizer(summarizer_model_name, text)
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

    def _call_summarizer(self, summarizer_model: str, text: str) -> str:
        if summarizer_model.endswith("gguf"):
            # TODO handle style
            summary, memories = llama.ask(
                summarizer_model, text, self._summary_memories
            )
        else:
            summary, memories = gemma.summarize(
                text, self._summary_style, self._summary_memories
            )
        self._summary_memories += memories
        return summary

    def _hallucinate_chunk(self, text: str, model: str) -> str:
        chunk_summary = text
        for i in range(self._hallucination_rounds_per_chunk):
            _write_to_log(
                self._log_file_name,
                "Hallucinating current round: "
                + str(i)
                + " model: "
                + model
                + "\n\n"
                + "Text: "
                + text,
            )
            if self._hallucinator_model_names != []:
                hallucinator_model_name = random.choice(self._hallucinator_model_names)
            else:
                hallucinator_model_name = "gemma.cpp"
            hallucinated_continuation = self._call_hallucinator(
                finnegannise(chunk_summary), hallucinator_model_name
            )
            _write_to_log(
                self._log_file_name,
                "\n\n\n----------------->Hallucination after round "
                + str(i)
                + " by model "
                + hallucinator_model_name
                + ": \n\n"
                + hallucinated_continuation,
            )
            chunk_summary = chunk_summary + " " + hallucinated_continuation
        return chunk_summary

    def _call_hallucinator(self, text: str, model: str) -> str:
        if model.endswith("gguf"):
            # TODO handle style
            hallucination = llama.ask(model, text, [])[0]
        else:
            hallucination = gemma.hallucinate(text, self._hallucination_style)
        return hallucination

    def _sentence_tokenizer(self, text: str) -> List[str]:
        return nltk.tokenize.sent_tokenize(text)

    def _divide_text(self, text: str) -> List[str]:
        # TODO: Use the tokenizer from the model
        tokenizer = AutoTokenizer.from_pretrained(picked_models.summarization_models[0])
        max_token_length = 1024

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

    def _create_out_filename(
        self, source_name: str, format: str = "txt", postfix: str = ""
    ) -> str:
        output_filename = (
            "summary_of_"
            + source_name.split("/")[-1]
            + "_"
            + postfix
            + "_at_"
            + str(self._creation_time)
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

    print("Summarizing text from path: " + src_file_path)

    summarizer_models = args.parse_args().summarizer_models
    hallucinator_models = args.parse_args().hallucinator_models
    hallucination_rounds_per_chunk = args.parse_args().hallucination_rounds_per_chunk
    summary_style = args.parse_args().summary_style
    hallucination_style = args.parse_args().hallucination_style

    narration_on = args.parse_args().narration_on
    summarization_rounds_per_chunk = args.parse_args().summarization_rounds_per_chunk
    src_file = open(
        src_file_path,
        "r",
    ).read()
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
