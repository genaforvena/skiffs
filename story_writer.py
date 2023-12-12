import models_to_consider
from text_utils import get_last_sentence
from runner import run_model

def _do_run_and_get_next_prompt(f, generator, prompt):
    try:
        f.write("Prompt: " + prompt + "\n")
        out = generator(
            prompt,
            do_sample=True,
            min_length=20,
            max_new_tokens=100,
        )
        out = out[0]["generated_text"]
        if out.startswith(prompt):
            print("Prompt: " + prompt)
            out = out[len(prompt) :]
        print(out)
        f.write(out + "\n")
        return get_last_sentence(out)
    except Exception as e:
        print(e)
        return "But never so failed. Worse failed. With care never worse failed."


if __name__ == "__main__":
    rounds = 100
    prompt = "Say for be said. Missaid. From now say for missaid."

    for model in models_to_consider.models:
        run_model(model, prompt, rounds, _do_run_and_get_next_prompt)
