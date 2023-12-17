from transformers import AutoTokenizer
from util import log
from text_continuator import generate_continuation
import subprocess
import sys


# Function to safely execute Python code and return the output
def _execute_python_code(code):
    try:
        # Execute the Python code in a separate process
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output, errors = process.communicate()
        return output, errors, process.returncode
    except Exception as e:
        return "", str(e), -1


def _extract_generated_code(code_gen_reply):
    # Just trying to be simple for now
    return code_gen_reply

def _get_model_max_length(tokenizer):
    return tokenizer.model_max_length

def _modify_prompt_based_on_feedback(previous_prompt, modification):
    new_prompt = f"{previous_prompt}\n\n# Modification:\n{modification}"
    return new_prompt

def _ensure_within_context_window(prompt, max_length, tokenizer):
    tokens = tokenizer.encode(prompt)
    if len(tokens) > max_length:
        return tokenizer.decode(tokens[:max_length])
    return prompt


def generate_code_by_adjusting_prompt(coding_model_name, prompt_gen_model_name, prompter_init_prompt):
    log("code_generation.txt", "\n\n")
    log("code_generation.txt", "Prompt: " + prompter_init_prompt)
    log("code_generation.txt", "Code generation model: " + coding_model_name + "\n")
    log("code_generation.txt", "Prompt generation model: " + prompt_gen_model_name + "\n")
    
    tokenizer = AutoTokenizer.from_pretrained(prompt_gen_model_name)
    prompt = prompter_init_prompt
    errors = ""
    # Loop until the return code is 0 (no error)
    while True:
        prompt = _modify_prompt_based_on_feedback(prompt, f"Please ensure to {errors}.", tokenizer)
        prompt = _ensure_within_context_window(current_prompt, max_length, tokenizer)
        prompt = generate_continuation(prompt_gen_model_name, prompt, 1)
        log("code_generation.txt", "Prompt to code generator: " + prompt + "\n")
        code_gen_reply = generate_continuation(coding_model_name, prompt, 1)
        log("code_generation.txt", "Code generation reply: " + code_gen_reply + "\n")
        generated_code = _extract_generated_code(code_gen_reply)
        log("code_generation.txt", "Generated code: " + generated_code + "\n")
        # Execute the Python code
        output, errors, returncode = _execute_python_code(generated_code)
        # Display results
        print("Generated Code:", generated_code)
        print("Output:", output)
        print("Errors:", errors)
        print("Return Code:", returncode)
        if returncode == 0:
            return generated_code 
    return "No code for you!"


if __name__ == "__main__":
    for prompt_gen_model_name in models_to_consider.prompt_generation_models:
        for code_gen_model_name in models_to_consider.code_generation_models:
            generate_code_by_adjusting_prompt(code_gen_model_name, prompt_gen_model_name, "Write a Python function to add two numbers")
