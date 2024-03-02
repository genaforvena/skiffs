import torch
import simpleaudio as sa
from transformers import VitsModel, AutoTokenizer


def narrate(msg: str) -> None:
    model = VitsModel.from_pretrained("facebook/mms-tts-eng")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

    inputs = tokenizer(msg, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs).waveform

    # Convert the waveform to a numpy array
    waveform = output.squeeze().numpy()
    # Normalize the waveform to 16-bit signed integers
    waveform_int16 = (waveform * 32767).astype("int16")

    # Play the audio
    play_obj = sa.play_buffer(waveform_int16, 1, 2, 22050)
    play_obj.wait_done()  # Adjust the rate if necessary
