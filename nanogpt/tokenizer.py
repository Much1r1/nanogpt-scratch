from typing import List, Union, Set

class CharTokenizer:
    def __init__(self, chars: Union[List[str], Set[str], None] = None):
        if chars is None:
            self.chars = []
        else:
            self.chars = sorted(list(set(chars)))

        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text: str) -> List[int]:
        encoded = []
        for ch in text:
            if ch not in self.stoi:
                raise ValueError(f"Character {ch!r} not in vocabulary")
            encoded.append(self.stoi[ch])
        return encoded

    def decode(self, tokens: List[int]) -> str:
        decoded = []
        for token in tokens:
            if token not in self.itos:
                raise ValueError(f"Token ID {token!r} not in vocabulary")
            decoded.append(self.itos[token])
        return "".join(decoded)

    @classmethod
    def from_corpus(cls, text: str) -> "CharTokenizer":
        chars = sorted(list(set(text)))
        return cls(chars)

    @classmethod
    def from_file(cls, filepath: str, encoding: str = "utf-8") -> "CharTokenizer":
        with open(filepath, "r", encoding=encoding) as f:
            text = f.read()
        return cls.from_corpus(text)
