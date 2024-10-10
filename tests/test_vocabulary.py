import pytest


def test_create_vocabulary():
    from nugpt import Vocabulary
    vocab = Vocabulary()
    assert vocab is not None

def test_create_special_tokens():
    from nugpt import Vocabulary
    vocab = Vocabulary()
    assert len(vocab.token2id) == 1
    assert "SPECIAL" in vocab.token2id
    assert len(vocab.token2id["SPECIAL"]) == len(vocab.special_tokens)

def test_add_new_tokens():
    from nugpt import Vocabulary
    vocab = Vocabulary()
    vocab.set_field_keys(["TEST"])
    assert vocab.set_id("test", "TEST") == 7
    assert len(vocab.token2id) == 2
    assert vocab.set_id("test", "TEST") == 7
    assert len(vocab.token2id) == 2
    assert vocab.set_id("test", "TEST", return_local=True) == 0
    assert vocab.set_id("test", "TEST", return_local=True) == 0
    assert vocab.set_id("test2", "TEST") == 8
    assert vocab.set_id("test2", "TEST", return_local=True) == 1

def test_get_local_from_global_ids():
    from nugpt import Vocabulary
    vocab = Vocabulary()
    vocab.set_field_keys(["TEST"])
    vocab.set_id("test", "TEST")
    vocab.set_id("test2", "TEST")
    assert vocab.get_local_from_global_ids([0, 1, 2, 3, 4, 5, 6, 7, 8]).tolist() == [0, 1, 2, 3, 4, 5, 6, 0, 1]