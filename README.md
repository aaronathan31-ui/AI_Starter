# DocuMind AI — Research Assistant (Starter)

Upload PDFs, search them using sentence embeddings, and optionally generate answers grounded in retrieved passages. All citations are shown as `[S1]` etc., with document names and 1-indexed page numbers.

## Setup (Python 3.11 recommended)

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The first model run downloads an embedding model. An internet connection is required for that initial download.

## Optional LLM answers

Copy `.env.example` to `.env`; add your own `OPENAI_API_KEY`. Set `OPENAI_MODEL` to a model available on your API account. Without a key the app still supports semantic retrieval and citations. API usage may incur charges; ChatGPT subscription and API billing are separate.

Never push `.env`, API secrets, copyrighted documents, or private datasets to GitHub. Streamlit hosting secrets should be configured through its secret manager; you may also set `OPENAI_API_KEY` as an environment variable in the deployment environment.

## Tests

```bash
python -m pytest -q
```

## Evaluate retrieval

Create a question CSV based on your own PDF using `eval/questions_template.csv` as a starting point. Each row must specify the correct PDF basename and expected page.

```bash
python -m eval.evaluate_retrieval data/sample_docs/example.pdf eval/questions.csv
```

The metric is page-level Hit@3. Avoid evaluating on questions used to tune the model. For multi-document evaluations, adapt the evaluation script to index all PDFs.

## Architecture

PDF -> pypdf -> overlapping word chunks with page metadata -> MiniLM embeddings -> cosine retrieval -> optional LLM generation -> cited source excerpts.

## Limitations

- Only text-based PDFs; no OCR or layout-aware table interpretation.
- Citations refer to retrieved passages; an LLM can still make unsupported claims or misuse source markers. Verify claims against excerpts.
- Retrieval scores are *not* calibrated probabilities.
- In-memory index and chat history reset on session restart. No login or persistent storage.
- Small, non-sensitive documents only; no production security audit.
- First model load can be slow on free hosting.

## Portfolio roadmap

1. Build a TF-IDF baseline, compare Hit@3 with embedding search.
2. Evaluate retrieval on 20–30 answerable and unanswerable questions.
3. Measure citation support rate and refuse unanswerable questions.
4. Improve chunking, filtering, caching, and add Docker/CI.
5. Deploy a public demo with sample permissively licensed PDFs.

## Resume (after completing the work)

Built a source-grounded PDF research assistant with Python, semantic embeddings, and Streamlit; measured retrieval Hit@3 on an annotated benchmark and deployed an interactive demo.
