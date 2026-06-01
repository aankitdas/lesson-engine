import re

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)         # remove wiki refs like [1]
    text = re.sub(r'<.*?>', '', text)           # strip leftover html
    text = re.sub(r'http\S+', '', text)         # strip urls
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # strip non-ascii
    return text.strip()

def word_count(text: str) -> int:
    return len(text.split())

def is_usable(text: str, min_words: int = 30, max_words: int = 600) -> bool:
    wc = word_count(text)
    return min_words <= wc <= max_words