"""
Expanded Dataset Evaluation — 100 Document Sets
Victor Chukwudi Robinson — MSc AI & Data Science, UEL 2026

Builds a SEPARATE vector store for the expanded 100-set corpus
and runs full evaluation, without touching the existing 20-set results.

Usage:
    python evaluate_expanded.py --build     # Build vector store (30-40 min)
    python evaluate_expanded.py --evaluate  # Run evaluation
    python evaluate_expanded.py --all       # Both
"""

import os
import sys
import json
import glob
import time
import pickle
import argparse
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, 'src')

import faiss
from openai import OpenAI
from ingestion_pipeline_v2 import process_document

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Separate paths for expanded dataset
EXPANDED_DIR = Path("vector_store_expanded")
EXPANDED_DIR.mkdir(exist_ok=True)

FAISS_PATH = EXPANDED_DIR / "faiss_index.bin"
METADATA_PATH = EXPANDED_DIR / "metadata.json"
DOCRESULTS_PATH = EXPANDED_DIR / "document_results.pkl"
EVAL_PATH = EXPANDED_DIR / "evaluation_results.json"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


# ============================================================
# BUILD VECTOR STORE
# ============================================================

def build_expanded_vector_store():
    """Build vector store from the 100-set expanded dataset."""
    
    print("="*60)
    print("BUILDING EXPANDED VECTOR STORE — 100 DOCUMENT SETS")
    print("="*60)
    
    pdf_files = sorted(glob.glob("data/expanded_dataset/**/*.pdf", recursive=True))
    print(f"\nFound {len(pdf_files)} PDF files")
    
    if len(pdf_files) == 0:
        print("ERROR: No PDFs found. Run generate_expanded_dataset.py first.")
        return
    
    # Process all documents
    print("\nProcessing documents through ingestion pipeline...")
    document_results = []
    failed = []
    
    for i, filepath in enumerate(pdf_files, 1):
        try:
            result = process_document(filepath)
            document_results.append(result)
            if i % 25 == 0:
                print(f"  Processed {i}/{len(pdf_files)} documents...")
        except Exception as e:
            failed.append((filepath, str(e)))
    
    print(f"\nProcessed: {len(document_results)} documents")
    if failed:
        print(f"Failed: {len(failed)} documents")
        for f, e in failed[:5]:
            print(f"  - {os.path.basename(f)}: {e}")
    
    # Inherit PO refs from folder siblings
    print("\nInheriting PO references from folder siblings...")
    folder_po_map = {}
    for result in document_results:
        if result.po_ref and result.po_ref not in ['NOT_FOUND', 'NOT_PROVIDED']:
            folder = os.path.dirname(result.source_file)
            if folder not in folder_po_map:
                folder_po_map[folder] = result.po_ref
    
    inherited = 0
    for result in document_results:
        if result.po_ref in ['NOT_FOUND', 'NOT_PROVIDED']:
            folder = os.path.dirname(result.source_file)
            if folder in folder_po_map:
                result.po_ref = folder_po_map[folder]
                for chunk in result.chunks:
                    if chunk.po_ref in ['NOT_FOUND', 'NOT_PROVIDED']:
                        chunk.po_ref = folder_po_map[folder]
                inherited += 1
    print(f"  Inherited PO refs for {inherited} documents")
    
    # Save document results
    with open(DOCRESULTS_PATH, 'wb') as f:
        pickle.dump(document_results, f)
    
    # Build embeddings
    total_chunks = sum(len(d.chunks) for d in document_results)
    print(f"\nBuilding vector store from {len(document_results)} documents, {total_chunks} chunks...")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print("-"*60)
    
    all_embeddings = []
    all_metadata = []
    chunk_counter = 0
    
    for doc_result in document_results:
        for chunk in doc_result.chunks:
            chunk_counter += 1
            
            # Build enriched text with metadata prefix
            parts = []
            if chunk.po_ref and chunk.po_ref != 'NOT_FOUND':
                parts.append(f"[{chunk.po_ref}]")
            parts.append(f"[{doc_result.document_type}]")
            parts.append(f"[{doc_result.supplier_format}]")
            parts.append(f"[{chunk.chunk_type}]")
            parts.append(chunk.content)
            enriched_text = " ".join(parts)
            
            # Get embedding
            try:
                response = client.embeddings.create(
                    input=enriched_text,
                    model=EMBEDDING_MODEL
                )
                embedding = response.data[0].embedding
                all_embeddings.append(embedding)
                
                all_metadata.append({
                    'filename': os.path.basename(doc_result.source_file),
                    'source_file': doc_result.source_file,
                    'document_type': doc_result.document_type,
                    'supplier_format': doc_result.supplier_format,
                    'po_ref': chunk.po_ref,
                    'invoice_ref': doc_result.invoice_ref,
                    'dn_ref': doc_result.dn_ref,
                    'chunk_type': chunk.chunk_type,
                    'content': chunk.content,
                    'doc_items': doc_result.items,
                    'doc_total': doc_result.total_amount,
                    'confidence': doc_result.confidence_score,
                    'hitl_required': doc_result.hitl_required,
                })
                
                if chunk_counter % 50 == 0:
                    print(f"  [{chunk_counter}/{total_chunks}] chunks embedded")
                    
            except Exception as e:
                print(f"  Error embedding chunk: {e}")
    
    print(f"\nTotal chunks embedded: {len(all_embeddings)}")
    
    # Build FAISS index
    print(f"\nBuilding FAISS index (IndexFlatL2, {EMBEDDING_DIM} dimensions)...")
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings_array)
    print(f"FAISS index built — {index.ntotal} vectors stored")
    
    # Save
    faiss.write_index(index, str(FAISS_PATH))
    with open(METADATA_PATH, 'w') as f:
        json.dump(all_metadata, f, indent=2, default=str)
    
    print(f"\nFAISS index saved to: {FAISS_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")
    print("\n" + "="*60)
    print("EXPANDED VECTOR STORE BUILD COMPLETE")
    print("="*60)
    
    return index, all_metadata, document_results


# ============================================================
# EVALUATION
# ============================================================

def evaluate_expanded():
    """Run full evaluation on the expanded 100-set corpus."""
    
    print("="*60)
    print("EVALUATING EXPANDED CORPUS — 100 DOCUMENT SETS")
    print("="*60)
    
    # Load expanded vector store
    if not FAISS_PATH.exists():
        print("ERROR: Expanded vector store not found. Run with --build first.")
        return
    
    index = faiss.read_index(str(FAISS_PATH))
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    with open(DOCRESULTS_PATH, 'rb') as f:
        document_results = pickle.load(f)
    
    print(f"\nLoaded: {index.ntotal} vectors, {len(metadata)} chunks, {len(document_results)} documents")
    
    # Load ground truth
    with open("data/expanded_dataset/ground_truth_expanded.json") as f:
        ground_truth = json.load(f)
    
    print(f"Ground truth: {len(ground_truth)} document sets")
    
    # Monkey-patch the agent module to use expanded vector store
    import agent as agent_module
    agent_module._index = index
    agent_module._metadata = metadata
    agent_module._document_results = document_results
    agent_module._anomaly_log = []
    
    from verification_pipeline import verify_document_set
    
    # Get all PO refs from ground truth
    po_refs = [gt['po_ref'] for gt in ground_truth]
    gt_map = {gt['po_ref']: gt for gt in ground_truth}
    
    print(f"\nRunning evaluation on {len(po_refs)} document sets...")
    print("="*60)
    
    results = []
    start_time = time.time()
    
    for i, po_ref in enumerate(po_refs, 1):
        gt = gt_map[po_ref]
        expected_anomaly = gt['has_anomaly']
        
        try:
            result = verify_document_set(po_ref, verbose=False)
            detected_anomaly = not result['consistent']
            anomaly_types = list(set(a['type'] for a in result['anomalies']))
            
            correct = (detected_anomaly == expected_anomaly)
            
            results.append({
                'set_num': gt['set_num'],
                'po_ref': po_ref,
                'supplier_format': gt['supplier_format'],
                'expected_anomaly_type': gt['anomaly_type'],
                'expected_has_anomaly': expected_anomaly,
                'detected_has_anomaly': detected_anomaly,
                'detected_types': anomaly_types,
                'correct': correct,
                'num_anomalies': len(result['anomalies']),
            })
            
            status = "✓" if correct else "✗"
            if i % 10 == 0 or not correct:
                print(f"  [{i:3d}/100] {po_ref} | {gt['anomaly_type']:20s} | "
                      f"{'ANOMALY' if detected_anomaly else 'CONSISTENT':10s} {status}")
                
        except Exception as e:
            print(f"  [{i:3d}/100] {po_ref} ERROR: {e}")
            results.append({
                'set_num': gt['set_num'],
                'po_ref': po_ref,
                'supplier_format': gt['supplier_format'],
                'expected_anomaly_type': gt['anomaly_type'],
                'expected_has_anomaly': expected_anomaly,
                'detected_has_anomaly': None,
                'detected_types': [],
                'correct': False,
                'error': str(e),
            })
    
    elapsed = time.time() - start_time
    
    # Calculate metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS — 100 DOCUMENT SETS")
    print("="*60)
    
    tp = sum(1 for r in results if r['expected_has_anomaly'] and r['detected_has_anomaly'])
    tn = sum(1 for r in results if not r['expected_has_anomaly'] and r['detected_has_anomaly'] == False)
    fp = sum(1 for r in results if not r['expected_has_anomaly'] and r['detected_has_anomaly'])
    fn = sum(1 for r in results if r['expected_has_anomaly'] and r['detected_has_anomaly'] == False)
    errors = sum(1 for r in results if r.get('detected_has_anomaly') is None)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(results)
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {tp}")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  Errors:          {errors}")
    
    print(f"\nMetrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    
    print(f"\nProcessing:")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Avg per set: {elapsed/len(results):.2f}s")
    
    # Per-category breakdown
    print(f"\nPer-Category Performance:")
    categories = {}
    for r in results:
        cat = r['expected_anomaly_type']
        if cat not in categories:
            categories[cat] = {'total': 0, 'correct': 0}
        categories[cat]['total'] += 1
        if r['correct']:
            categories[cat]['correct'] += 1
    
    for cat, stats in sorted(categories.items()):
        pct = stats['correct'] / stats['total'] * 100
        print(f"  {cat:22s}: {stats['correct']:2d}/{stats['total']:2d} ({pct:5.1f}%)")
    
    # Per-format breakdown
    print(f"\nPer-Format Performance:")
    formats = {}
    for r in results:
        fmt = r['supplier_format']
        if fmt not in formats:
            formats[fmt] = {'total': 0, 'correct': 0}
        formats[fmt]['total'] += 1
        if r['correct']:
            formats[fmt]['correct'] += 1
    
    for fmt, stats in sorted(formats.items()):
        pct = stats['correct'] / stats['total'] * 100
        print(f"  {fmt:22s}: {stats['correct']:2d}/{stats['total']:2d} ({pct:5.1f}%)")
    
    # Save results
    output = {
        'metrics': {
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'errors': errors,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
        },
        'per_category': categories,
        'per_format': formats,
        'processing': {
            'total_seconds': elapsed,
            'avg_per_set': elapsed / len(results),
        },
        'num_sets': len(results),
        'results': results,
    }
    
    with open(EVAL_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {EVAL_PATH}")
    print("\nFor dissertation reporting:")
    print(f"  Extended evaluation on 100 document sets achieved F1 = {f1:.3f}")
    print(f"  with precision = {precision:.3f} and recall = {recall:.3f}")
    
    return output


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true', help='Build expanded vector store')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation')
    parser.add_argument('--all', action='store_true', help='Build and evaluate')
    args = parser.parse_args()
    
    if args.all or args.build:
        build_expanded_vector_store()
    
    if args.all or args.evaluate:
        evaluate_expanded()
    
    if not any([args.build, args.evaluate, args.all]):
        print("Usage:")
        print("  python evaluate_expanded.py --build     # Build vector store (~30 min)")
        print("  python evaluate_expanded.py --evaluate  # Run evaluation")
        print("  python evaluate_expanded.py --all       # Both")
