from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "PygmalionAI/pygmalion-350m"
tokenizer_path = "PygmalionAI/pygmalion-350m"


# Load the model and tokenizer


model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Your input text here
input_text = "What is the meaning of the concept body-wthiout-organs?"



# Encode the input text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate a response
output = model.generate(input_ids, max_new_tokens=100)



# Decode the output
print(tokenizer.decode(output[0], skip_special_tokens=True))
