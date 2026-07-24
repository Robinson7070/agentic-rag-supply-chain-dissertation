"""
Traditional RAG Baseline — for comparison with Agentic RAG System
Victor Chukwudi Robinson — MSc AI & Data Science, UEL 2026

Implements single-step retrieve-then-generate RAG (Lewis et al., 2020):
- No tool use
- No multi-step reasoning  
- Single retrieval then single LLM call
- No HITL
- No anomaly logging
"""

import os, json, time, sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
sys.path.insert(0, 'src')

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4o"
TOP_K = 5


@dataclass
class RAGResult:
    query: str
    retrieved_chunks: List[Dict]
    llm_response: str
    execution_time: float
    tokens_used: int
    chunks_retrieved: int
    detected_anomaly: bool


def load_vector_store():
    import faiss
    index = faiss.read_index("vector_store/faiss_index.bin")
    with open("vector_store/metadata.json") as f:
        metadata = json.load(f)
    return index, metadata


def retrieve_chunks(query: str, index, metadata: List[Dict],
                    top_k: int = TOP_K, po_ref: Optional[str] = None) -> List[Dict]:
    """Single-step retrieval — no iteration, no tools."""
    response = client.embeddings.create(input=query, model=EMBEDDING_MODEL)
    query_vector = np.array([response.data[0].embedding], dtype=np.float32)
    distances, indices = index.search(query_vector, top_k * 10)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        chunk = metadata[idx].copy()
        chunk['similarity_score'] = float(1 / (1 + dist))
        if po_ref and chunk.get('po_ref') != po_ref:
            continue
        results.append(chunk)
    
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return results[:top_k]


def generate_response(query: str, chunks: List[Dict]) -> tuple:
    """Single LLM call — no tools, no structured verification."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Document {i}: {chunk.get('filename','unknown')} | "
            f"Type: {chunk.get('document_type','unknown')} | "
            f"Section: {chunk.get('chunk_type','unknown')}]\n"
            f"{chunk.get('content','')}\n"
            f"Items: {json.dumps(chunk.get('doc_items',[]))}"
        )
    
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""You are a supply chain document auditor.
Using ONLY the document excerpts provided below, answer the question.
Do not use any external knowledge. Base your answer entirely on the retrieved context.

RETRIEVED DOCUMENTS:
{context}

QUESTION: {query}

State clearly: CONSISTENT or ANOMALY DETECTED. If anomaly, describe it specifically."""

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": "You are a supply chain document auditor. Answer based only on provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=800
    )
    
    return response.choices[0].message.content, response.usage.total_tokens


def traditional_rag_query(query: str, po_ref: Optional[str] = None) -> RAGResult:
    """Full traditional RAG: single retrieve + single generate."""
    start = time.time()
    index, metadata = load_vector_store()
    chunks = retrieve_chunks(query, index, metadata, top_k=TOP_K, po_ref=po_ref)
    llm_response, tokens = generate_response(query, chunks)
    elapsed = time.time() - start
    
    anomaly_keywords = ['mismatch', 'discrepancy', 'differ', 'inconsist',
                       'anomaly', 'overcharge', 'incorrect', 'wrong', 'missing']
    detected = any(kw in llm_response.lower() for kw in anomaly_keywords)
    
    return RAGResult(
        query=query,
        retrieved_chunks=chunks,
        llm_response=llm_response,
        execution_time=round(elapsed, 2),
        tokens_used=tokens,
        chunks_retrieved=len(chunks),
        detected_anomaly=detected
    )


# Ground truth for test cases
GROUND_TRUTH = {
    "PO-2026-2001": {"has_anomaly": True,  "type": "price_mismatch",    "detail": "VP-204: PO £42.00 vs Invoice £46.50"},
    "PO-2026-2002": {"has_anomaly": True,  "type": "price_mismatch",    "detail": "FS-118: PO £165.00 vs Invoice £178.00"},
    "PO-2026-2004": {"has_anomaly": True,  "type": "quantity_mismatch", "detail": "EC-902: ordered 18 delivered 15"},
    "PO-2026-1001": {"has_anomaly": False, "type": None,                "detail": "Consistent"},
    "PO-2026-1002": {"has_anomaly": False, "type": None,                "detail": "Consistent"},
    "PO-2026-2007": {"has_anomaly": True,  "type": "missing_document",  "detail": "Delivery note missing"},
}


def run_comparison_evaluation():
    """Run Traditional RAG on 6 test cases and save results."""
    print("\n" + "="*60)
    print("TRADITIONAL RAG BASELINE EVALUATION")
    print("="*60)

    test_cases = [
        {"po_ref": "PO-2026-2001", "query": "Does the invoice for PO-2026-2001 match the purchase order? Check unit prices and quantities."},
        {"po_ref": "PO-2026-2002", "query": "Verify the invoice for PO-2026-2002 matches the purchase order in unit prices."},
        {"po_ref": "PO-2026-2004", "query": "Does the delivery note for PO-2026-2004 match the ordered quantities?"},
        {"po_ref": "PO-2026-1001", "query": "Are all documents for PO-2026-1001 consistent with no discrepancies?"},
        {"po_ref": "PO-2026-1002", "query": "Verify the invoice for PO-2026-1002 matches the purchase order."},
        {"po_ref": "PO-2026-2007", "query": "Are all three documents present and consistent for PO-2026-2007?"},
    ]

    results = []

    for tc in test_cases:
        po_ref = tc["po_ref"]
        gt = GROUND_TRUTH.get(po_ref, {})
        
        print(f"\n{po_ref} (Expected: {'ANOMALY' if gt.get('has_anomaly') else 'CONSISTENT'})")
        
        result = traditional_rag_query(tc["query"], po_ref=po_ref)
        correct = (result.detected_anomaly == gt.get('has_anomaly', False))
        
        print(f"  Detected: {'ANOMALY' if result.detected_anomaly else 'CONSISTENT'} {'✅' if correct else '❌'}")
        print(f"  Time: {result.execution_time}s | Tokens: {result.tokens_used}")
        print(f"  Response: {result.llm_response[:150]}...")
        
        results.append({
            "po_ref": po_ref,
            "ground_truth_anomaly": gt.get('has_anomaly'),
            "ground_truth_type": gt.get('type'),
            "ground_truth_detail": gt.get('detail'),
            "trad_detected": result.detected_anomaly,
            "trad_correct": correct,
            "trad_time": result.execution_time,
            "trad_tokens": result.tokens_used,
            "trad_chunks": result.chunks_retrieved,
            "trad_response": result.llm_response,
        })

    # Save results
    with open("vector_store/comparison_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    correct_count = sum(1 for r in results if r['trad_correct'])
    total = len(results)
    avg_time = sum(r['trad_time'] for r in results) / total
    avg_tokens = sum(r['trad_tokens'] for r in results) / total

    print("\n" + "="*60)
    print("TRADITIONAL RAG RESULTS")
    print("="*60)
    print(f"Accuracy:      {correct_count}/{total} = {correct_count/total*100:.0f}%")
    print(f"Avg time:      {avg_time:.2f}s per query")
    print(f"Avg tokens:    {avg_tokens:.0f} per query")
    print()
    print("AGENTIC RAG RESULTS (from full evaluation):")
    print(f"Accuracy:      19/20 = 95% (F1: 0.974)")
    print(f"Precision:     100% (zero false positives)")
    print(f"Missing docs:  Detected")
    print(f"Date checks:   Detected")
    print(f"HITL:          Yes")
    print(f"Traceability:  Yes")
    print()
    print("Results saved to vector_store/comparison_results.json")
    
    return results


if __name__ == "__main__":
    run_comparison_evaluation()
