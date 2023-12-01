import os


def load_latest_checkpoint(logs_folder, version=None):
    if version is None:
        version_directories = [d for d in os.listdir(logs_folder) if 'version' in d]
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
