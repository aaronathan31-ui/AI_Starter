"""Evaluate retrieval Hit@k on a manually authored question set.
Usage: python -m eval.evaluate_retrieval path/to/your.pdf eval/questions.csv
"""
import csv
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.pipeline import extract_pdf, SearchIndex


def main(pdf_path, questions_path, k=3):
    pdf = Path(pdf_path)
    index = SearchIndex(extract_pdf(pdf.read_bytes(), pdf.name), SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"))
    with open(questions_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Evaluation CSV is empty")
    hits = 0
    for row in rows:
        found = any(c.source == row["expected_source"] and c.page == int(row["expected_page"])
                    for c, _ in index.search(row["question"], top_k=k))
        hits += found
        print(("HIT" if found else "MISS") + " | " + row["question"])
    print(f"Hit@{k}: {hits}/{len(rows)} = {hits / len(rows):.1%}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m eval.evaluate_retrieval example.pdf eval/questions.csv")
    main(sys.argv[1], sys.argv[2])
