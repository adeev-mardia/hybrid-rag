#!/usr/bin/env python3
"""Minimal end-to-end usage example.

Usage:
    python scripts/demo.py
"""

from hybrid_rag.document import Document
from hybrid_rag.eval_data import load_corpus
from hybrid_rag.pipeline import HybridRetriever


def main() -> None:
    corpus = load_corpus()
    retriever = HybridRetriever(corpus)

    query = "why can't python run threads in parallel"
    print(f"Query: {query!r}\n")

    for mode in ("bm25", "dense", "hybrid"):
        print(f"--- mode={mode} ---")
        for rank, doc in enumerate(retriever.search(query, top_k=3, mode=mode), start=1):
            print(f"{rank}. [{doc.doc_id}] (score={doc.score:.3f}) {doc.text}")
        print()


if __name__ == "__main__":
    main()
