import pytest
from nanogpt.tokenizer import CharTokenizer

def test_vocab_generation_from_corpus():
    corpus = "hello world!"
    tokenizer = CharTokenizer.from_corpus(corpus)

    expected_chars = sorted(list(set(corpus)))
    assert tokenizer.chars == expected_chars
    assert tokenizer.vocab_size == len(expected_chars)
    assert len(tokenizer.stoi) == len(expected_chars)
    assert len(tokenizer.itos) == len(expected_chars)

def test_vocab_generation_from_file(tmp_path):
    filepath = tmp_path / "sample.txt"
    corpus = "Testing from_file method."
    filepath.write_text(corpus, encoding="utf-8")

    tokenizer = CharTokenizer.from_file(str(filepath))
    expected_chars = sorted(list(set(corpus)))
    assert tokenizer.chars == expected_chars
    assert tokenizer.vocab_size == len(expected_chars)

def test_encode_decode_roundtrip():
    corpus = "abcdefghijklmnopqrstuvwxyz 0123456789!"
    tokenizer = CharTokenizer.from_corpus(corpus)

    original_text = "hello 123!"
    tokens = tokenizer.encode(original_text)
    decoded_text = tokenizer.decode(tokens)

    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)
    assert decoded_text == original_text

def test_unknown_character_error():
    tokenizer = CharTokenizer.from_corpus("abc")

    with pytest.raises((ValueError, KeyError)):
        tokenizer.encode("abcd")

def test_unknown_token_id_error():
    tokenizer = CharTokenizer.from_corpus("abc")

    with pytest.raises((ValueError, KeyError)):
        tokenizer.decode([999])
