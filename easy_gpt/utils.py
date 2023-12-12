import os
import json
import re


def load_latest_checkpoint(logs_folder, version=None):
    if version is None:
        version_directories = [d for d in os.listdir(logs_folder) if "version" in d]
        latest_version = max(version_directories, key=lambda x: int(x.split("_")[1]))
    else:
        latest_version = version
    checkpoints_path = f"{logs_folder}/{latest_version}/checkpoints"
    ckpt_file_name = os.listdir(checkpoints_path)[0]
    return os.path.abspath(f"{checkpoints_path}/{ckpt_file_name}")


def remove_unicode_chars(text: str) -> str:
    """
    Remove unicode characters still present in the read file (I could not resolve with a difference encoding somehow)

    Args:
        text: the read text file

    Returns:
        the cleaned text file
    """
    replacements = [
        ("\u0080", ""),
        ("\u0093", ""),
        ("\u0098", ""),
        ("\u0099", ""),
        ("\u009c", ""),
        ("\u009d", ""),
        ("\u00a6", ""),
        ("\u00e2", ""),
        ("\u00fc", ""),
    ]

    for char, replacement in replacements:
        if char in text:
            text = text.replace(char, replacement)

    return text


def clean(text):
    text = re.sub(r"[^A-Za-z0-9 \-\.;,\n\?!]", '', text)
    text = re.sub("\n+", " ", text)
    text = re.sub(" +", " ", text)
    return text


def txt_to_json(file_path):
    # Read the text file and split lines
    with open(file_path, "r") as file:
        lines = file.read().splitlines()

    # Define a regex pattern for extracting questions and answers
    pattern = r"\+\+\+(.+?);---(.+)"

    # Extract questions and answers using regex
    matches = [re.match(pattern, line) for line in lines]
    matches = [match.groups() if match else (None, None) for match in matches]

    # Separate the questions and answers
    prompts, responses = zip(
        *[(q, a) for q, a in matches if q is not None and a is not None]
    )

    # Convert the list of questions to a dictionary with 'question' as the key
    data_dict = {"question": list(prompts), "answer": list(responses)}

    json_string = json.dumps(data_dict)
    # Specify the file path
    file_path = "data/harry_finetuning.json"

    # Write the JSON string to the file
    with open(file_path, "w", encoding="utf-8") as writer:
        writer.write(json_string)
