import os
import torch

OPENAI_API_KEY = ""
OPENAI_BASE_URL = ""

# huggingface target models
ROOT_PATH = "PATH/TO/MODELS/ROOT"
VICUNA_7B_PATH = os.path.join(ROOT_PATH, "lmsys/vicuna-7b-v1.5")
VICUNA_13B_PATH = os.path.join(ROOT_PATH, "lmsys/vicuna-13b-v1.5")
LLAMA2_PATH = os.path.join(ROOT_PATH, "meta-llama/Llama-2-7b-chat-hf")
LLAMA3_PATH = os.path.join(ROOT_PATH, "meta-llama/Meta-Llama-3.1-8B-Instruct")

# judge model
DEBERTA_PATH = "PATH/TO/CLASSIFIER/OUTPUT_DIR"
