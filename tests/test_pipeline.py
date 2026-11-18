from src.pipeline import Chunk, chunk_text, SearchIndex
import numpy as np


def test_chunks_keep_page_metadata():
    result = chunk_text("one two three four five six", "paper.pdf", 7, chunk_words=4, overlap_words=1)
    assert result[0] == Chunk("paper.pdf", 7, "one two three four")
    assert result[1].text == "four five six"


def test_empty_text():
    assert chunk_text("  ", "empty.pdf", 1) == []


class FakeModel:
    def encode(self, texts, **kwargs):
        return np.array([[1., 0.] if "cat" in text else [0., 1.] for text in texts])


def test_semantic_retrieval_ranks_expected_chunk():
    index = SearchIndex([Chunk("a.pdf", 1, "cat facts"), Chunk("b.pdf", 2, "dog facts")], FakeModel())
    assert index.search("cat", top_k=1)[0][0].source == "a.pdf"
