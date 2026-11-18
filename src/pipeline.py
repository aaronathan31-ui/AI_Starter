"""Minimal, inspectable retrieval pipeline with page-level PDF citations."""
import io
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from pypdf import PdfReader


@dataclass(frozen=True)
class Chunk:
    source: str
    page: int
    text: str


def chunk_text(text: str, source: str, page: int, chunk_words: int = 160, overlap_words: int = 30) -> list[Chunk]:
    if chunk_words <= 0 or overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("Require chunk_words > overlap_words >= 0")
    words = re.sub(r"\s+", " ", text).strip().split()
    step = chunk_words - overlap_words
    return [Chunk(source, page, " ".join(words[i:i + chunk_words]))
            for i in range(0, len(words), step) if words[i:i + chunk_words]]


def extract_pdf(file_bytes: bytes, filename: str) -> list[Chunk]:
    if not file_bytes.startswith(b"%PDF"):
        raise ValueError(f"{filename}: Not a recognizable PDF file")
    reader = PdfReader(io.BytesIO(file_bytes))
    if reader.is_encrypted:
        raise ValueError(f"{filename}: Password-protected PDFs are not supported")
    chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        chunks.extend(chunk_text(text, filename, page_num))
    return chunks


class SearchIndex:
    def __init__(self, chunks: list[Chunk], model):
        if not chunks:
            raise ValueError("No extractable text found; scanned/image PDFs need OCR")
        self.chunks = chunks
        self.model = model
        self.vectors = model.encode([c.text for c in chunks], normalize_embeddings=True, convert_to_numpy=True)

    def search(self, question: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        if not question.strip():
            return []
        query_vector = self.model.encode([question], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = self.vectors @ query_vector
        indices = np.argsort(scores)[::-1][:max(1, min(top_k, len(self.chunks)))]
        return [(self.chunks[i], float(scores[i])) for i in indices]
