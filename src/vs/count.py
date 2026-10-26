import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Counts and returns the number of token present within a body of text."""
    return len(ENC.encode(text=text))
