"""
Utility functions called by the main instruction model script.
"""
import csv
import string
from typing import Optional

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import anthropic

EXAMPLES = """
Le président de la FIFA Sepp Blatter rejette les accusations des manifestants en les accusant d’opportunisme. → Le président de la FIFA Sepp Blatter rejette les accusations de la manifestation en l’accusant d'opportunisme.
Les auteurs et les spectateurs ont été satisfaits des réponses des représentants. → L'autorat et le public ont été satisfaits des réponses de la représentation.
Le vicaire général proposa de disperser les religieux dans d'autres maisons de l'ordre et de procéder à la réfection des bâtiments. → Le vicaire général proposa de disperser le couvent dans d'autres maisons de l'ordre et de procéder à la réfection des bâtiments.
"""

EXAMPLES_CORR = """
e président de la FIFA Sepp Blatter rejette les accusations de la manifestation en les accusant d’opportunisme. → Le président de la FIFA Sepp Blatter rejette les accusations de la manifestation en l’accusant d'opportunisme.
L'autorat et le public a été satisfaits des réponses des la représentation. → L'autorat et le public ont été satisfaits des réponses de la représentation.
Le vicaire générale proposa de disperser le couvent dans d'autres maisons de l'ordre et de procéder à la réfection des bâtiments. → Le vicaire général proposa de disperser le couvent dans d'autres maisons de l'ordre et de procéder à la réfection des bâtiments.
"""

MAX_TOKENS_MULT = 2.3

def load_sents(inp_file: str, prompt_type: str) -> list[str]:
    """
    Load sentences from a CSV file based on the specified prompt type.

    Args:
        inp_file (str): The path to the input CSV file.
        prompt_type (str): The type of prompt to determine which row to extract from the CSV.

    Returns:
        list: A list of sentences extracted from the CSV file.
    """
    sents = []

    if prompt_type in ["lazy", "dict"]:
        col_number = 1 # non_incl_sent
    elif prompt_type == "correction":
        col_number = 2 # rbs_auto_incl_sent
    else:
        raise ValueError("Unknown prompt_type", prompt_type)

    with open(inp_file, "r", encoding="utf8") as inp:
        reader = csv.reader(inp, delimiter=",")
        next(reader)
        data = list(reader)

        for row in data:
            sents.append(row[col_number])

    print(f"Loaded {len(sents)} sentences")
    return sents

def get_dict_data(sent: str) -> tuple[list[str], list[str]]:
    """
    Retrieves member nouns and collective nouns from a given sentence.

    Args:
        sent (str): The input sentence.

    Returns:
        tuple: A tuple containing two lists - member_nouns and collective_nouns.
               member_nouns: A list of member nouns extracted from the sentence.
               collective_nouns: A list of corresponding collective nouns for the member nouns.
    """
    tokens = [token.strip(string.punctuation) for token in sent.split()]
    member_nouns = []
    collective_nouns = []

    with open("../data/collective_nouns/collective_nouns.csv", "r", encoding="utf8") as cn_file:
        reader = csv.reader(cn_file, delimiter=",")
        next(reader)

        for row in reader:
            for token in tokens:
                if (token.lower() == row[3]) and (member_nouns.count(token) < tokens.count(token)):
                    member_nouns.append(token.lower())
                    collective_nouns.append(row[1])

    return member_nouns, collective_nouns

def create_prompt(prompt_type: str, sent: str,
                  member_nouns: Optional[list]=None,
                  collective_nouns: Optional[list]=None,
                  few_shots=False) -> str:
    """
    Creates a prompt based on the given parameters.

    Args:
        prompt_type (str): The type of prompt to create (lazy, dict, correction).
        sent (str): The original sentence to be included in the prompt.
        member_nouns (list[str], optional): List of generic masculine nouns to be replaced.
        Defaults to None.
        collective_nouns (list[str], optional): List of French collective noun equivalents
        for the member nouns. Defaults to None.
        few_shots (bool, optional): Flag indicating whether to generate
        a few-shot prompt. Defaults to False.

    Returns:
        str: The generated prompt.
    """
    prompt = ""
    if prompt_type == "lazy":
        if few_shots:
            prompt += f"Make this French sentence inclusive by replacing generic masculine nouns with their French collective noun equivalents. Generate the final sentence only without any comments nor notes.\n{EXAMPLES}\n{sent} → "
        else:
            prompt += f"Make this French sentence inclusive by replacing generic masculine nouns with their French collective noun equivalents. Generate the final sentence only without any comments nor notes: {sent}"

    elif prompt_type == "dict":
        if member_nouns is None or collective_nouns is None:
            raise ValueError("member_nouns or collective_nouns is None")

        if len(member_nouns) == 0 and len(collective_nouns) == 0:
            raise ValueError(f"Empty member_nouns or collective_nouns\n{member_nouns=}\n{collective_nouns=}")
        if len(member_nouns) != len(collective_nouns):
            raise ValueError("member_nouns and collective_nouns have different lengths")

        if len(member_nouns) >= 2:
            if few_shots:
                prompt += f"Make this French sentence inclusive by replacing generic masculine nouns \"{', '.join(member_nouns)}\" with their respective French collective noun equivalents \"{', '.join(collective_nouns)}\". Generate the final sentence only without any comments nor notes.\n{EXAMPLES}\n{sent} → "
            else:
                prompt += f"Make this French sentence inclusive by replacing generic masculine nouns \"{', '.join(member_nouns)}\" with their respective French collective noun equivalents \"{', '.join(collective_nouns)}\". Generate the final sentence only without any comments nor notes: {sent}"
        else:
            if few_shots:
                prompt += f"Make this French sentence inclusive by replacing generic masculine noun \"{''.join(member_nouns)}\" with its French collective noun equivalent \"{''.join(collective_nouns)}\". Generate the final sentence only without any comments nor notes.\n{EXAMPLES}\n{sent} →"
            else:
                prompt += f"Make this French sentence inclusive by replacing generic masculine noun \"{''.join(member_nouns)}\" with its French collective noun equivalent \"{''.join(collective_nouns)}\". Generate the final sentence only without any comments nor notes: {sent}"

    elif prompt_type == "correction":
        if few_shots:
            prompt += f"Correct grammar in this French sentence. Generate the final sentence only without any comments nor notes: {sent}"
        else:
            prompt += f"Correct grammar in this French sentence. Generate the final sentence only without any comments nor notes\n{EXAMPLES_CORR}\n{sent}"
    else:
        raise ValueError("Unknown prompt_type", prompt_type)

    return prompt

def request_mistral(message: str, sent: str, api_key: str) -> str:
    """
    Sends a request to the Mistral API and returns the response.

    Args:
        message (str): The message to send to the Mistral API.
        sent (str): The original sentence (used for calculating the max tokens parameter).
        api_key (str): The API key.

    Returns:
        str: The response from the Mistral chatbot.
    """
    client = MistralClient(api_key=api_key)

    messages = [
        ChatMessage(role="user", content=message)
    ]

    chat_response = client.chat(
        model="mistral-small", # Mixtral 8x7B
        messages=messages,
        random_seed="29751",
        max_tokens=int(round(len(sent.split())*MAX_TOKENS_MULT, 0)),
        temperature=0.4
    )

    return chat_response.choices[0].message.content

def request_claude(message: str, api_key: str) -> str:
    """
    Sends a request to the Claude API and returns the response.

    Args:
        message (str): The input message to send to the Claude API.
        api_key (str): The API key.

    Returns:
        str: The response from the Claude API.
    """
    client = anthropic.Anthropic(
        api_key=api_key,
    )
    message = client.messages.create(
        model="claude-3-opus-20240229",
        temperature=0,
        messages=[
            {"role": "user", "content": f"{message}"},
            {"role": "assistant", "content": "Here is the output sentence:"}
        ]
    )

    return message.content
