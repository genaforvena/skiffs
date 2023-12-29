from transformers import AutoModelForCausalLM, AutoTokenizer


def mamba_reply(text):
    model = AutoModelForCausalLM.from_pretrained(
        "Q-bert/Mamba-1B", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Q-bert/Mamba-1B")
    input_ids = tokenizer.encode(text, return_tensors="pt")
    output = model.generate(
        input_ids, max_length=20, num_beams=5, no_repeat_ngram_size=2
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)
