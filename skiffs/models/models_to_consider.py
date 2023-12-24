conversation_models = [
    # "KoboldAI/OPT-350M-Erebus", too pornographic
    # "EleutherAI/gpt-neo-125M",
    # "microsoft/DialoGPT-small",
    # "microsoft/DialoGPT-medium",
    # "facebook/blenderbot-400M-distill",
    # "microsoft/DialoGPT-large",
    # "facebook/blenderbot_small-90M",
    # "ToddGoldfarb/Cadet-Tiny",
    # "mywateriswet/ShuanBot",
    # "PygmalionAI/pygmalion-350m",
    # "satvikag/chatbot",
    # "tinkoff-ai/ruDialoGPT-medium",
    "gpt2",
]  # "gpt-neo-125m",

object_detectors = ["microsoft/table-transformer-structure-recognition-v1.1-all"]

info_exttractors = ["microsoft/layoutlm-base-uncased"]

sql_interpreters = ["microsoft/tapex-large-sql-execution"]

image_to_text_converters = ["microsoft/git-base-vatex"]

text_classiriers = [
    "distilbert-base-uncased-finetuned-sst-2-english",
    "roberta-large-mnli",
]

text_transforers = ["sentence-transformers/all-mpnet-base-v2"]

text_comparators = ["sentence-transformers/all-mpnet-base-v2", "clips/mfaq"]

question_answerers = [
    "deepset/roberta-base-squad2",
    "google/tapas-base-finetuned-wtq",  # this one might be sql interpreter. not sure
]

evaluator_models = [
    "khalidrizk/autotrain-mental-health-503641204167",
]

text_continuators = [
    # "KoboldAI/OPT-350M-Erebus",
    # "PygmalionAI/pygmalion-350m",
    # "bigscience/bloomz-560m",
    # "cmarkea/bloomz-560m-sft-chat",
    # "L-R/LLmRa-1.3B",
    # "ericzzz/falcon-rw-1b-chat",
    # "deepseek-ai/deepseek-coder-1.3b-instruct",
    "gpt2",
    # "distilgpt2",
    # "facebook/opt-1.3b",
    # "HuggingFaceM4/tiny-random-LlamaForCausalLM",
    # "EleutherAI/gpt-neo-125m",
    "facebook/opt-125M",
    # "facebook/opt-350m",
    # "microsoft/phi-1",
    # "DarwinAnim8or/GPT-Greentext-355m"
]

summarization_models = [
    "sshleifer/distilbart-cnn-12-6",
    # "facebook/bart-large-cnn",
    # "SamAct/PromptGeneration-base",
    "Falconsai/medical_summarization",
]

reward_models = [
    "nicholasKluge/RewardModel",
]

code_generation_models = [
    "gpt2",
    "gpt-neo-125m",
]

instructuct_text_generation_models = [
    "nicholasKluge/Aira-2-124M-DPO",
]

text_expansion_models = [
    "divg07/facebook-bart-large-news",
    "Cohee/bart-factbook-summarization",
]

code_explanation_models = ["ashwinR/CodeExplainer"]

prompt_generation_models = ["gpt2", "gpt-neo-125m", "SamAct/PromptGeneration-base"]
