"""
Tools to generate README files with Markdown.
"""

from datetime import datetime


def save_params(path: str, func_name: str, msg: str = None, **params) -> None:
    """
    Generate Markdown file with an introduction message and a list of parameters
    as a list.
    """
    now = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M")
    buffer = f"Parameters of '{func_name}' as of {now}.\n\n"
    if msg is not None:
        buffer += f"{msg}\n\n"
    for k, v in params.items():
        if v is not None:
            str_v = f"'{v}'" if type(v) == str else v
            buffer += f" - {k}: {str_v}\n"
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(buffer)
