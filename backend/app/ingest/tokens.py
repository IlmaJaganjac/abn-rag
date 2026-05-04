from __future__ import annotations

import tiktoken

EMBEDDING_MAX_TOKENS = 8191
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text, disallowed_special=()))


def split_oversize(text: str, max_tokens: int, overlap: int) -> list[str]:
    tokens = _ENCODING.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return [text]
    if overlap >= max_tokens:
        overlap = max_tokens // 4
    step = max_tokens - overlap
    pieces: list[str] = []
    start = 0
    while start < len(tokens):
        window = tokens[start : start + max_tokens]
        pieces.append(_ENCODING.decode(window))
        if start + max_tokens >= len(tokens):
            break
        start += step
    return pieces
