"""
Embeddings and Vector Store Layer — Version 1
Agentic RAG System for Automated Document Intelligence in Supply Chain Operations

This module converts structure-aware document chunks into vector embeddings
and stores them in a FAISS vector database with enriched metadata.

Design decisions informed by literature:
- text-embedding-3-small for cost-effective, high-quality embeddings
- FAISS IndexFlatL2 for exact nearest-neighbour search at our scale (56 docs)
- Metadata stored separately alongside FAISS index for hybrid filtering
- Chunk-level embedding (not document-level) for precise retrieval
- Metadata-aware filtering before semantic search — Cheerla (2025)
- Structure-aware chunks prevent semantic entanglement — Loghmani (2026)
"""

import os
import json
import pickle
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialise OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Embedding model — text-embedding-3-small gives 1536 dimensions
# Cost-effective and sufficient for our document scale
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# Paths
VECTOR_STORE_DIR = Path("vector_store")
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss_index.bin"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.json"
DOCRESULTS_PATH = VECTOR_STORE_DIR / "document_results.pkl"


# ============================================================
# EMBEDDING GENERATION
# ============================================================

def generate_embedding(text: str, retries: int = 3) -> List[float]:
    """
    Generate an embedding vector for a text string using OpenAI.
    
    Includes retry logic for API rate limits and transient errors.
    Each chunk is embedded separately to preserve semantic coherence —
    this directly implements the disentanglement approach of Loghmani (2026)
    where header, line_item, and summary chunks remain separate embeddings
    rather than being mixed into a single document-level embedding.
    """
    for attempt in range(retries):
        try:
            response = client.embeddings.create(
                input=text,
                model=EMBEDDING_MODEL
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt  # exponential backoff
                print(f"    Embedding error (attempt {attempt + 1}): {e}. "
                      f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e


def prepare_chunk_text_for_embedding(chunk) -> str:
    """
    Prepare the text representation of a chunk for embedding.
    
    Following Shah et al. (2026) structured metadata enrichment approach:
    prepend structured metadata tags to the chunk content before embedding.
    This enriches the vector representation with document context,
    improving retrieval accuracy for comparative queries (PO vs Invoice).
    
    Example output:
    "[purchase_order][meridian][PO-2026-1001][line_item]
     VP-204 Brake Pads 50 units £42.00 each"
    """
    # Build metadata prefix
    parts = []
    
    if chunk.document_type and chunk.document_type != 'unknown':
        parts.append(f"[{chunk.document_type}]")
    
    if chunk.supplier_format and chunk.supplier_format != 'unknown':
        parts.append(f"[{chunk.supplier_format}]")
    
    if chunk.po_ref and chunk.po_ref != 'NOT_FOUND':
        parts.append(f"[{chunk.po_ref}]")
    
    if chunk.invoice_ref and chunk.invoice_ref != 'NOT_FOUND':
        parts.append(f"[{chunk.invoice_ref}]")
    
    if chunk.dn_ref and chunk.dn_ref != 'NOT_FOUND':
        parts.append(f"[{chunk.dn_ref}]")
    
    parts.append(f"[{chunk.chunk_type}]")
    
    metadata_prefix = " ".join(parts)
    
    # Combine prefix with chunk content
    enriched_text = f"{metadata_prefix}\n{chunk.content}"
    
    return enriched_text


# ============================================================
# VECTOR STORE BUILDING
# ============================================================

def build_vector_store(document_results: list) -> tuple:
    """
    Build a FAISS vector store from a list of DocumentResult objects.
    
    Returns (faiss_index, metadata_list) where metadata_list contains
    the chunk metadata for every vector in the index, aligned by index position.
    
    Using FAISS IndexFlatL2 — exact search rather than approximate.
    At our scale (56 documents, ~300-400 chunks total) exact search
    is fast enough and avoids approximation errors that could cause
    the agent to miss relevant chunks during anomaly detection.
    """
    VECTOR_STORE_DIR.mkdir(exist_ok=True)
    
    all_embeddings = []
    all_metadata = []
    
    total_chunks = sum(len(dr.chunks) for dr in document_results)
    print(f"\nBuilding vector store from {len(document_results)} documents, "
          f"{total_chunks} chunks total...")
    print(f"Embedding model: {EMBEDDING_MODEL} ({EMBEDDING_DIM} dimensions)")
    print("-" * 60)
    
    chunk_count = 0
    
    for doc_result in document_results:
        filename = Path(doc_result.source_file).name
        print(f"\nProcessing: {filename} "
              f"({len(doc_result.chunks)} chunks, "
              f"confidence: {doc_result.confidence_score:.2f})")
        
        for chunk in doc_result.chunks:
            # Skip very short chunks that won't embed meaningfully
            if len(chunk.content.strip()) < 10:
                continue
            
            # Prepare enriched text for embedding
            enriched_text = prepare_chunk_text_for_embedding(chunk)
            
            # Generate embedding
            embedding = generate_embedding(enriched_text)
            all_embeddings.append(embedding)
            
            # Store metadata aligned with embedding index position
            chunk_metadata = {
                # Document-level metadata for filtering
                "source_file": chunk.source_file,
                "filename": filename,
                "document_type": chunk.document_type,
                "supplier_format": chunk.supplier_format,
                "po_ref": chunk.po_ref,
                "invoice_ref": chunk.invoice_ref,
                "dn_ref": chunk.dn_ref,
                
                # Chunk-level metadata
                "chunk_type": chunk.chunk_type,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "confidence": chunk.confidence,
                
                # Document-level extraction results
                "doc_total_amount": doc_result.total_amount,
                "doc_items": doc_result.items,
                "doc_confidence": doc_result.confidence_score,
                "doc_hitl_required": doc_result.hitl_required,
                "doc_warnings": doc_result.extraction_warnings,
                
                # Vector store position
                "vector_id": chunk_count
            }
            all_metadata.append(chunk_metadata)
            
            chunk_count += 1
            print(f"  [{chunk_count}/{total_chunks}] "
                  f"{chunk.chunk_type} chunk embedded "
                  f"(confidence: {chunk.confidence:.2f})")
            
            # Small delay to respect API rate limits
            time.sleep(0.05)
    
    print(f"\n{'-' * 60}")
    print(f"Total chunks embedded: {chunk_count}")
    
    # Build FAISS index
    print(f"\nBuilding FAISS index (IndexFlatL2, {EMBEDDING_DIM} dimensions)...")
    embeddings_matrix = np.array(all_embeddings, dtype=np.float32)
    
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings_matrix)
    
    print(f"FAISS index built — {index.ntotal} vectors stored")
    
    # Save index and metadata
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"FAISS index saved to: {FAISS_INDEX_PATH}")
    
    with open(METADATA_PATH, 'w') as f:
        json.dump(all_metadata, f, indent=2, default=str)
    print(f"Metadata saved to: {METADATA_PATH}")
    
    return index, all_metadata


# ============================================================
# VECTOR STORE LOADING
# ============================================================

def load_vector_store() -> tuple:
    """
    Load an existing FAISS index and metadata from disk.
    Call this instead of build_vector_store() after first build.
    """
    if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Vector store not found. Run build_vector_store() first."
        )
    
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    print(f"Vector store loaded — {index.ntotal} vectors, "
          f"{len(metadata)} metadata entries")
    
    return index, metadata


# ============================================================
# RETRIEVAL — METADATA-AWARE SEARCH
# Following Cheerla (2025) metadata-aware filtering approach
# ============================================================

def search_chunks(
    query: str,
    index: faiss.Index,
    metadata: List[Dict],
    top_k: int = 5,
    filter_document_type: Optional[str] = None,
    filter_po_ref: Optional[str] = None,
    filter_invoice_ref: Optional[str] = None,
    filter_dn_ref: Optional[str] = None,
    filter_chunk_type: Optional[str] = None,
    filter_supplier: Optional[str] = None,
) -> List[Dict]:
    """
    Search the vector store for chunks relevant to a query.
    
    Implements metadata-aware filtering following Cheerla (2025):
    hard filters on document reference numbers are applied AFTER
    semantic search, returning only results that match both
    semantic similarity AND structural criteria.
    
    This is critical for the agent's tools — when searching for
    "the invoice for PO-2026-2001", we want to filter by po_ref
    first to ensure we only retrieve chunks from that specific
    document set, not semantically similar chunks from other sets.
    
    Args:
        query: Natural language or structured query
        index: FAISS index
        metadata: Metadata list aligned with index
        top_k: Number of results to return after filtering
        filter_document_type: 'purchase_order', 'invoice', 'delivery_note'
        filter_po_ref: Specific PO reference number
        filter_invoice_ref: Specific invoice reference number
        filter_dn_ref: Specific delivery note reference number
        filter_chunk_type: 'header', 'line_item', 'summary'
        filter_supplier: 'meridian', 'castlegate', 'northgate', 'vantage', 'coastal'
    
    Returns:
        List of matching metadata dicts with added 'similarity_score' field
    """
    # Generate query embedding
    query_embedding = generate_embedding(query)
    query_vector = np.array([query_embedding], dtype=np.float32)
    
    # Search FAISS — retrieve more than top_k to allow for filtering
    search_k = min(len(metadata), top_k * 10)
    distances, indices = index.search(query_vector, search_k)
    
    # Collect results with scores
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:  # FAISS returns -1 for empty slots
            continue
        
        chunk_meta = metadata[idx].copy()
        # Convert L2 distance to similarity score (lower distance = higher similarity)
        chunk_meta['similarity_score'] = float(1 / (1 + dist))
        chunk_meta['l2_distance'] = float(dist)
        
        results.append(chunk_meta)
    
    # Apply metadata filters
    if filter_document_type:
        results = [r for r in results
                   if r.get('document_type') == filter_document_type]
    
    if filter_po_ref:
        results = [r for r in results
                   if r.get('po_ref') == filter_po_ref]
    
    if filter_invoice_ref:
        results = [r for r in results
                   if r.get('invoice_ref') == filter_invoice_ref]
    
    if filter_dn_ref:
        results = [r for r in results
                   if r.get('dn_ref') == filter_dn_ref]
    
    if filter_chunk_type:
        results = [r for r in results
                   if r.get('chunk_type') == filter_chunk_type]
    
    if filter_supplier:
        results = [r for r in results
                   if r.get('supplier_format') == filter_supplier]
    
    # Return top_k after filtering, sorted by similarity
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return results[:top_k]


def get_document_chunks(
    po_ref: str,
    index: faiss.Index,
    metadata: List[Dict],
    document_type: Optional[str] = None
) -> List[Dict]:
    """
    Retrieve all chunks for a specific PO reference number.
    
    This is the primary retrieval function for the agent's
    search_documents() tool — given a PO reference, retrieve
    all related document chunks (PO + Invoice + DN) for that
    transaction set.
    """
    results = [m.copy() for m in metadata if m.get('po_ref') == po_ref]
    
    if document_type:
        results = [r for r in results
                   if r.get('document_type') == document_type]
    
    # Sort by document type then chunk index for logical reading order
    type_order = {'purchase_order': 0, 'invoice': 1, 'delivery_note': 2, 'unknown': 3}
    results.sort(key=lambda x: (
        type_order.get(x.get('document_type', 'unknown'), 3),
        x.get('chunk_index', 0)
    ))
    
    return results


# ============================================================
# MAIN — BUILD THE VECTOR STORE
# ============================================================

if __name__ == "__main__":
    import sys
    import glob
    sys.path.insert(0, 'src')
    
    from ingestion_pipeline_v2 import process_document
    
    # Find all PDF files in the dataset
    pdf_files = sorted(glob.glob("data/**/*.pdf", recursive=True))
    
    if not pdf_files:
        print("No PDF files found. Make sure you're running from the repo root.")
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} PDF files")
    print("Processing all documents through ingestion pipeline...")
    print("=" * 60)
    
    # Process all documents
    document_results = []
    failed = []
    
    # First pass: process all documents
    for filepath in pdf_files:
        try:
            result = process_document(filepath)
            document_results.append(result)
            status = "HITL" if result.hitl_required else "OK"
            print(f"  [{status}] {filepath} "
                  f"(type: {result.document_type}, "
                  f"confidence: {result.confidence_score:.2f})")
        except Exception as e:
            print(f"  [ERROR] {filepath}: {e}")
            failed.append(filepath)
    
    # Second pass: inherit PO ref from sibling documents in same folder
    # This fixes DNs and Invoices that don't explicitly mention the PO number
    print("\nInheriting PO references from folder siblings...")
    
    # Build folder -> po_ref mapping from documents that DO have PO refs
    import os
    folder_po_map = {}
    for result in document_results:
        if result.po_ref and result.po_ref != 'NOT_FOUND' and not result.po_ref.startswith('PO-(informal'):
            folder = os.path.dirname(result.source_file)
            if folder not in folder_po_map:
                folder_po_map[folder] = result.po_ref
    
    # Apply inherited PO refs to documents with NOT_FOUND
    inherited_count = 0
    for result in document_results:
        if result.po_ref == 'NOT_FOUND' or result.po_ref.startswith('PO-(informal'):
            folder = os.path.dirname(result.source_file)
            if folder in folder_po_map:
                inherited_po = folder_po_map[folder]
                print(f"  Inheriting {inherited_po} for {os.path.basename(result.source_file)}")
                result.po_ref = inherited_po
                # Also update all chunks
                for chunk in result.chunks:
                    if chunk.po_ref == 'NOT_FOUND' or chunk.po_ref.startswith('PO-(informal'):
                        chunk.po_ref = inherited_po
                inherited_count += 1
    
    print(f"  Inherited PO refs for {inherited_count} documents")
    
    print(f"\nProcessed: {len(document_results)} documents")
    if failed:
        print(f"Failed: {len(failed)} documents")
        for f in failed:
            print(f"  - {f}")
    
    # Save document results for later use by agent
    VECTOR_STORE_DIR.mkdir(exist_ok=True)
    with open(DOCRESULTS_PATH, 'wb') as f:
        pickle.dump(document_results, f)
    print(f"\nDocument results saved to: {DOCRESULTS_PATH}")
    
    # Build vector store
    index, metadata = build_vector_store(document_results)
    
    print("\n" + "=" * 60)
    print("VECTOR STORE BUILD COMPLETE")
    print("=" * 60)
    
    # Quick verification search
    print("\nRunning verification search...")
    test_query = "price mismatch invoice purchase order VP-204"
    results = search_chunks(
        query=test_query,
        index=index,
        metadata=metadata,
        top_k=3,
        filter_chunk_type='line_item'
    )
    
    print(f"\nTop 3 results for: '{test_query}'")
    for i, r in enumerate(results, 1):
        print(f"\n  Result {i}:")
        print(f"    File: {r['filename']}")
        print(f"    Type: {r['document_type']} | Chunk: {r['chunk_type']}")
        print(f"    PO Ref: {r['po_ref']}")
        print(f"    Similarity: {r['similarity_score']:.4f}")
        print(f"    Content preview: {r['content'][:100]}...")
    
    print("\nVector store ready for agent layer.")
