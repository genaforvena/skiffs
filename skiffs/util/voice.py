import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import simpleaudio as sa


def narrate(msg: str) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler_tts_mini_v0.1"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")

    description = "A male speaker with a very low-pitched voice delivers his words with zero emotions, in a very confined sounding environment with clear audio quality. He speaks quite slowly."

    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(msg, return_tensors="pt").input_ids.to(device)

    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    waveform = generation.cpu().numpy().squeeze()
    waveform_int16 = (waveform * 32767).astype("int16")

    play_obj = sa.play_buffer(waveform_int16, 1, 2, 22050)
    play_obj.wait_done()
