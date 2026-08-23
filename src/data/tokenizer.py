import tiktoken
import torch

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

def text_to_token_ids(text, tokenizer, device):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    decoded = tokenizer.decode(flat.tolist())
    return decoded