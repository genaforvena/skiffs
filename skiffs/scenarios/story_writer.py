import models_to_consider
from text_utils import get_last_sentence
from text_continuator import generate_continuation

def _do_run_and_get_next_prompt(f, prev_prompt, model_out):
    try:
        out = model_out
        if model_out.startswith(prev_prompt):
            print("Prompt: " + prev_prompt)
            out = out[len(prev_prompt) :]
        print(out)
        f.write(out + "\n")
        return get_last_sentence(out)
    except Exception as e:
        print(e)
        return "But never so failed. Worse failed. With care never worse failed."


if __name__ == "__main__":
    rounds = 100
    prompt = "Say for be said. Missaid. From now say for missaid."

    for model in models_to_consider.text_continuators:
        generate_continuation(model, prompt, rounds, _do_run_and_get_next_prompt)
