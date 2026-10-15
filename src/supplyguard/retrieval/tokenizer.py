import re

TOKEN_PATTERN = re.compile(r"[@a-zA-Z0-9][@a-zA-Z0-9._/+:-]*")

def tokenize(text):
    if not text:
        return []

    raw_tokens = TOKEN_PATTERN.findall(text.lower())
    tokens = []

    for token in raw_tokens:
        cleaned_token = token.strip("._/+:-")

        if cleaned_token:
            tokens.append(cleaned_token)

    return tokens