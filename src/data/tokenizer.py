import tiktoken
from typing import Final

_CACHE: dict[str, tiktoken.Encoding] = {}


def get_tokenizer(name: str = "gpt2") -> tiktoken.Encoding:
    if name not in _CACHE:
        _CACHE[name] = tiktoken.get_encoding(name)
    return _CACHE[name]


def vocab_size(name: str = "gpt2") -> int:
    enc = get_tokenizer(name)
    return enc.n_vocab


def encode(text: str, name: str = "gpt2") -> list[int]:
    return get_tokenizer(name).encode(text)


def decode(ids: list[int], name: str = "gpt2") -> str:
    return get_tokenizer(name).decode(ids)
