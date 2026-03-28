from tokenizer.vocabulary import Vocabulary


def test_vocabulary_encode_decode_and_persistence(tmp_path):
    vocab = Vocabulary(max_size=32)
    vocab.build([["x", "+", "1"], ["sin", "(", "x", ")"]])

    ids = vocab.encode(["x", "+", "1"], add_boundaries=True)
    decoded = vocab.decode(ids, remove_boundaries=True)
    assert decoded == ["x", "+", "1"]

    path = tmp_path / "vocab.json"
    vocab.save(str(path))
    loaded = Vocabulary.load(str(path))
    assert loaded.token_to_idx == vocab.token_to_idx
    assert loaded.idx_to_token == vocab.idx_to_token
