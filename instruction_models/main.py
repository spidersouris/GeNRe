from configparser import ConfigParser
from utils import load_sents, create_prompt, get_dict_data, request_claude, request_mistral

results = []
SENT_COUNT = 0

config = ConfigParser()
config.read("config.ini")

try:
    MODEL_TYPE = config["model_type"]
    PROMPT_TYPE = config["prompt_type"]
    FEW_SHOTS = config.getboolean("few_shots")
    INP_FILE = config["input_file"]
    OUT_FILE = config["output_file"]

    if not all([MODEL_TYPE, PROMPT_TYPE, FEW_SHOTS, INP_FILE, OUT_FILE]):
        raise ValueError("Missing parameter in config.ini")
except KeyError as e:
    raise ValueError(f"Missing key {e} in config.ini") from e

if MODEL_TYPE == "mistral":
    api_key = config["mistral_api_key"]
elif MODEL_TYPE == "claude":
    api_key = config["claude_api_key"]
else:
    raise ValueError("Unknown model type", MODEL_TYPE)

if api_key in ["", "YOUR_API_KEY"]:
    raise ValueError(f"Set API key for {MODEL_TYPE} in config.ini")

sents = load_sents(INP_FILE, PROMPT_TYPE)

for sent in sents:
    if PROMPT_TYPE == "dict":
        member_nouns, collective_nouns = get_dict_data(sent)
        prompt = create_prompt("dict", sent, member_nouns, collective_nouns, few_shots=FEW_SHOTS)
    else:
        prompt = create_prompt(PROMPT_TYPE, sent, few_shots=FEW_SHOTS)

    SENT_COUNT += 1
    print("Generating sentence", SENT_COUNT)

    if MODEL_TYPE == "mistral":
        results.append(request_mistral(prompt, sent, api_key))
    elif MODEL_TYPE == "claude":
        results.append(request_claude(prompt, api_key))
