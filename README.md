# Agentic RAG System for Automated Document Intelligence in Supply Chain Operations

An agentic Retrieval-Augmented Generation system that detects anomalies across linked supply chain documents — purchase orders, invoices, delivery notes, and goods received notes — by reasoning across multiple documents rather than answering from a single retrieved chunk.

Built as an MSc dissertation project (AI & Data Science, University of East London), grounded in real-world logistics document intelligence problems.

## Results

Evaluated on a 56-document corpus spanning 20 document sets, each either internally consistent or containing one of five anomaly types (price discrepancy, quantity mismatch, missing document, date conflict, supplier mismatch):

| Metric | Score |
|---|---|
| Anomaly type identification — F1 | **0.974** (precision 1.000, recall 0.950) |
| Binary anomaly detection — F1 | 1.000 |
| Hop Completeness (novel metric, below) | 0.975 |
| Cohen's Kappa vs. ground truth | 1.000 |

Against a traditional single-hop RAG baseline (flat retrieval, no cross-document reasoning), the agentic system detected anomalies at **95% accuracy vs. 67%** for the baseline, with the gap concentrated entirely in cases requiring evidence from more than one document.

On an extended 100-set corpus, precision held at 1.000 but recall fell to 0.573 — degradation traced entirely to the regex-based field extraction layer rather than the retrieval or reasoning components. Full analysis in the dissertation writeup.

## Why agentic, not standard RAG

Standard RAG retrieves a chunk and answers from it. Most supply chain anomalies aren't visible in any single document — a price discrepancy only exists in the *relationship* between a PO's agreed price and an invoice's billed price. This system uses an agent loop that plans which documents to pull, retrieves across them, and verifies its own conclusions before returning an answer, rather than a single retrieve-then-generate pass.

## Architecture

```
Ingestion  →  Embedding  →  Agent (plan / retrieve / reason)  →  Verification
   │              │                    │                            │
   ▼              ▼                    ▼                            ▼
structure-    FAISS vector      multi-hop retrieval          cross-checks the
aware         store             across linked docs           agent's own
chunking                                                     conclusion
```

- **`ingestion_pipeline.py`** — parses PDFs (via `pdfplumber`) and applies structure-aware chunking: documents are split by their internal structure (e.g. header / line-items / summary for an invoice) rather than generic fixed-length windows, so a chunk preserves complete semantic units instead of cutting across them arbitrarily.
- **`embeddings_store.py`** — embeds chunks and builds the FAISS vector index.
- **`agent.py`** — the agentic reasoning layer: plans retrieval across potentially multiple linked documents, forms an answer.
- **`verification_pipeline.py`** — checks the agent's conclusion against the retrieved evidence before it's returned.
- **`traditional_rag.py`** — single-hop RAG baseline used for comparison.
- **`ragas_manual.py`** — custom RAGAS-style evaluation (faithfulness, answer relevancy, context precision/recall) plus the novel **Hop Completeness** metric, which scores whether an answer actually drew on evidence from every document it needed to, rather than just producing a plausible-sounding response.
- **`statistical_significance.py`** — bootstrap confidence intervals, McNemar's test, and Cohen's Kappa for the agentic-vs-baseline comparison.

## Human-in-the-loop

Two intervention points rather than one: mid-retrieval, when the agent's confidence in a retrieved document is low, and post-verification, when the verification layer flags a conclusion as anomalous. This keeps a human in the loop exactly where the system is least certain, rather than reviewing everything or nothing.

## Stack

Python · LangChain/LangGraph · OpenAI GPT-4o · FAISS · pdfplumber · Streamlit (evaluation dashboard)

## Running it

```bash
git clone https://github.com/Robinson7070/agentic-rag-supply-chain-dissertation.git
cd agentic-rag-supply-chain-dissertation
python -m venv venv311
venv311\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file:
```
OPENAI_API_KEY=your_key_here
```

Ingest and build the vector store, then run the agent:
```bash
python src/ingestion_pipeline.py
python src/embeddings_store.py
python src/agent.py
```

Run the evaluation suite:
```bash
python src/ragas_manual.py
python src/statistical_significance.py
```

## Status

This is dissertation research code, shared for portfolio purposes. A conference paper based on the structure-aware chunking approach and the Hop Completeness metric is in preparation — implementation detail beyond what's described above is intentionally withheld pending submission.

## Author

Victor Chukwudi Robinson — [GitHub](https://github.com/Robinson7070) · MSc AI & Data Science, UEL
