import re

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters. No stemming, no stopwords."""
    return _TOKEN.findall(text.lower())
