"""
Agent Orchestration Layer — Version 1
Agentic RAG System for Automated Document Intelligence in Supply Chain Operations

This module implements the LangChain ReAct agent with five custom tools
for cross-document supply chain anomaly detection.

Architecture decisions informed by literature:
- ReAct framework (Yao et al., 2023) — Thought/Action/Observation loop
- Five discrete tools with non-overlapping functions — Xu et al. (2025)
- LangChain as LLM orchestrator — Shen et al. (2023) paradigm
- Dual HITL intervention points — Lazaros et al. (2026), Adedokun et al. (2026)
- Memory layer for context persistence — Liu et al. (2024)
"""

import os
import json
import pickle
from pathlib import Path
from typing import Optional
from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

load_dotenv()

# ============================================================
# LOAD VECTOR STORE
# ============================================================

VECTOR_STORE_DIR = Path("vector_store")
METADATA_PATH = VECTOR_STORE_DIR / "metadata.json"
DOCRESULTS_PATH = VECTOR_STORE_DIR / "document_results.pkl"
ANOMALY_LOG_PATH = VECTOR_STORE_DIR / "anomaly_log.json"

def load_resources():
    """Load vector store index, metadata, and document results."""
    import faiss
    import numpy as np
    
    faiss_path = VECTOR_STORE_DIR / "faiss_index.bin"
    if not faiss_path.exists():
        raise FileNotFoundError(
            "Vector store not found. Run embeddings_store.py first."
        )
    
    index = faiss.read_index(str(faiss_path))
    
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    with open(DOCRESULTS_PATH, 'rb') as f:
        document_results = pickle.load(f)
    
    print(f"Vector store loaded: {index.ntotal} vectors, "
          f"{len(metadata)} chunks, {len(document_results)} documents")
    
    return index, metadata, document_results


# Global resources — loaded once at startup
_index = None
_metadata = None
_document_results = None
_anomaly_log = []


def get_resources():
    """Get or initialise global resources."""
    global _index, _metadata, _document_results
    if _index is None:
        _index, _metadata, _document_results = load_resources()
    return _index, _metadata, _document_results


# ============================================================
# TOOL 1: search_documents
# Primary retrieval tool — uses vector store with metadata filtering
# ============================================================

@tool
def search_documents(query: str) -> str:
    """
    Search supply chain documents in the vector store.
    
    Use this tool to retrieve document chunks relevant to a query.
    You can include reference numbers in the query for more precise retrieval.
    
    Examples:
    - "PO-2026-2001 purchase order line items"
    - "INV-MV-8102 invoice total amount"
    - "DN-9921 delivery note items received"
    - "VP-204 price unit cost"
    
    Returns the most relevant document chunks with their metadata.
    """
    import faiss
    import numpy as np
    from openai import OpenAI
    
    index, metadata, _ = get_resources()
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Generate query embedding
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_vector = np.array([response.data[0].embedding], dtype=np.float32)
    
    # Search FAISS
    distances, indices = index.search(query_vector, 20)
    
    # Extract PO/Invoice/DN references from query for filtering
    import re
    po_filter = re.search(r'PO-\d{4}-\d{3,4}', query)
    inv_filter = re.search(r'INV-[A-Z]{2,3}-\d{3,4}', query)
    dn_filter = re.search(r'DN-\d{3,4}', query)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        chunk = metadata[idx].copy()
        chunk['similarity_score'] = float(1 / (1 + dist))
        
        # Apply reference filters if found in query
        if po_filter and chunk.get('po_ref') != po_filter.group(0):
            continue
        if inv_filter and chunk.get('invoice_ref') != inv_filter.group(0):
            continue
        if dn_filter and chunk.get('dn_ref') != dn_filter.group(0):
            continue
        
        results.append(chunk)
    
    # Sort by similarity and take top 5
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    results = results[:5]
    
    if not results:
        return "No relevant documents found for this query."
    
    # Format results for the agent
    output_parts = []
    for i, r in enumerate(results, 1):
        part = (
            f"[Result {i}]\n"
            f"File: {r.get('filename', 'unknown')}\n"
            f"Document Type: {r.get('document_type', 'unknown')}\n"
            f"Supplier Format: {r.get('supplier_format', 'unknown')}\n"
            f"PO Ref: {r.get('po_ref', 'NOT_FOUND')}\n"
            f"Invoice Ref: {r.get('invoice_ref', 'NOT_FOUND')}\n"
            f"DN Ref: {r.get('dn_ref', 'NOT_FOUND')}\n"
            f"Chunk Type: {r.get('chunk_type', 'unknown')}\n"
            f"Content: {r.get('content', '')}\n"
            f"Document Total: £{r.get('doc_total_amount', 'N/A')}\n"
            f"Line Items: {json.dumps(r.get('doc_items', []), default=str)}\n"
            f"Confidence: {r.get('doc_confidence', 'N/A')}\n"
            f"HITL Required: {r.get('doc_hitl_required', False)}"
        )
        output_parts.append(part)
    
    return "\n\n---\n\n".join(output_parts)


# ============================================================
# TOOL 2: compare_values
# Compares two extracted values and identifies discrepancies
# ============================================================

@tool
def compare_values(comparison_input: str) -> str:
    """
    Compare two values extracted from different supply chain documents.
    
    Input format: "field_name|value1|value2|document1|document2"
    
    Examples:
    - "unit_price|42.00|46.50|PO-2026-2001|INV-MV-8102"
    - "quantity|40|35|PO-2026-2004|DN-9215"
    - "total_amount|3540.00|3720.00|PO-2026-1001|INV-MV-7741"
    
    Returns whether values match or a discrepancy report.
    """
    try:
        parts = comparison_input.split('|')
        if len(parts) != 5:
            return (
                "Invalid input format. Use: "
                "field_name|value1|value2|document1|document2"
            )
        
        field_name, val1_str, val2_str, doc1, doc2 = parts
        
        # Try numeric comparison first
        try:
            val1 = float(val1_str.replace('£', '').replace(',', '').strip())
            val2 = float(val2_str.replace('£', '').replace(',', '').strip())
            
            if abs(val1 - val2) < 0.001:  # floating point tolerance
                return (
                    f"MATCH: {field_name} is consistent across documents.\n"
                    f"{doc1}: {val1}\n"
                    f"{doc2}: {val2}\n"
                    f"No discrepancy detected."
                )
            else:
                difference = val2 - val1
                pct_change = (difference / val1) * 100 if val1 != 0 else 0
                direction = "OVERCHARGE" if difference > 0 else "UNDERCHARGE"
                
                return (
                    f"DISCREPANCY DETECTED: {field_name} does not match.\n"
                    f"{doc1}: £{val1:.2f}\n"
                    f"{doc2}: £{val2:.2f}\n"
                    f"Difference: £{abs(difference):.2f} "
                    f"({abs(pct_change):.1f}% {direction})\n"
                    f"Action required: Flag for human review."
                )
        
        except ValueError:
            # String comparison for non-numeric fields
            val1_clean = val1_str.strip().lower()
            val2_clean = val2_str.strip().lower()
            
            if val1_clean == val2_clean:
                return (
                    f"MATCH: {field_name} is consistent.\n"
                    f"{doc1}: {val1_str}\n"
                    f"{doc2}: {val2_str}"
                )
            else:
                return (
                    f"DISCREPANCY DETECTED: {field_name} does not match.\n"
                    f"{doc1}: {val1_str}\n"
                    f"{doc2}: {val2_str}\n"
                    f"Action required: Flag for human review."
                )
    
    except Exception as e:
        return f"Error in comparison: {str(e)}"


# ============================================================
# TOOL 3: calculate_total
# Calculates expected total from line items and compares to stated total
# ============================================================

@tool
def calculate_total(calculation_input: str) -> str:
    """
    Calculate the expected total from line items and compare to stated total.
    
    Input format: JSON string with items and stated total.
    Example: {"items": [{"code": "VP-204", "qty": 40, "unit_price": 42.00},
                         {"code": "VP-311", "qty": 100, "unit_price": 8.50}],
              "stated_total": 3540.00,
              "vat_rate": 0.20}
    
    Returns calculated total, VAT breakdown, and whether it matches stated total.
    """
    try:
        data = json.loads(calculation_input)
        items = data.get('items', [])
        stated_total = data.get('stated_total')
        vat_rate = data.get('vat_rate', 0.20)  # default UK VAT 20%
        
        if not items:
            return "No items provided for calculation."
        
        # Calculate subtotal from line items
        subtotal = 0.0
        line_details = []
        
        for item in items:
            code = item.get('code', 'UNKNOWN')
            qty = float(item.get('qty', 0))
            unit_price = float(item.get('unit_price', 0))
            line_total = qty * unit_price
            subtotal += line_total
            line_details.append(
                f"  {code}: {qty} units × £{unit_price:.2f} = £{line_total:.2f}"
            )
        
        vat_amount = subtotal * vat_rate
        calculated_total = subtotal + vat_amount
        
        result_parts = [
            "CALCULATION RESULT:",
            "Line items:",
        ]
        result_parts.extend(line_details)
        result_parts.append(f"Subtotal: £{subtotal:.2f}")
        result_parts.append(f"VAT ({vat_rate*100:.0f}%): £{vat_amount:.2f}")
        result_parts.append(f"Calculated Total: £{calculated_total:.2f}")
        
        if stated_total is not None:
            stated_total = float(stated_total)
            difference = abs(calculated_total - stated_total)
            
            if difference < 0.01:
                result_parts.append(
                    f"Stated Total: £{stated_total:.2f} — MATCHES calculated total."
                )
            else:
                result_parts.append(
                    f"Stated Total: £{stated_total:.2f} — "
                    f"DISCREPANCY of £{difference:.2f} detected."
                )
        
        return "\n".join(result_parts)
    
    except json.JSONDecodeError:
        return "Invalid JSON input. Please provide valid JSON."
    except Exception as e:
        return f"Calculation error: {str(e)}"


# ============================================================
# TOOL 4: check_document_exists
# Verifies whether a required document exists in the vector store
# ============================================================

@tool
def check_document_exists(check_input: str) -> str:
    """
    Check whether a specific document exists in the vector store.
    
    Use this to verify all three documents in a set are present
    before attempting cross-document comparison.
    
    Input format: "po_ref|document_type"
    Document types: purchase_order, invoice, delivery_note
    
    Examples:
    - "PO-2026-2001|invoice"
    - "PO-2026-1001|delivery_note"
    - "PO-2026-2004|purchase_order"
    """
    try:
        parts = check_input.split('|')
        if len(parts) != 2:
            return "Invalid input. Use: po_ref|document_type"
        
        po_ref, doc_type = parts[0].strip(), parts[1].strip()
        _, metadata, _ = get_resources()
        
        # Search metadata for matching documents
        matches = [
            m for m in metadata
            if m.get('po_ref') == po_ref
            and m.get('document_type') == doc_type
        ]
        
        if matches:
            filenames = list(set(m.get('filename', 'unknown') for m in matches))
            return (
                f"FOUND: {doc_type} for {po_ref} exists in vector store.\n"
                f"Files: {', '.join(filenames)}\n"
                f"Chunks available: {len(matches)}"
            )
        else:
            return (
                f"NOT FOUND: No {doc_type} found for {po_ref}.\n"
                f"This may indicate a missing document anomaly.\n"
                f"Action required: Flag as missing document."
            )
    
    except Exception as e:
        return f"Error checking document: {str(e)}"


# ============================================================
# TOOL 5: flag_anomaly
# Records detected anomalies for HITL review
# Second HITL intervention point — post-verification
# Following Lazaros et al. (2026) exception-handling design
# ============================================================

@tool
def flag_anomaly(anomaly_input: str) -> str:
    """
    Flag a detected anomaly for human review.
    
    This is the post-verification HITL intervention point.
    All flagged anomalies are logged and require human confirmation
    before being included in the final anomaly report.
    
    Input format: JSON string with anomaly details.
    Example: {"anomaly_type": "price_mismatch",
              "po_ref": "PO-2026-2001",
              "document1": "PO-2026-2001.pdf",
              "document2": "INV-MV-8102.pdf",
              "field": "unit_price",
              "value1": "42.00",
              "value2": "46.50",
              "impact": "£180.00 overcharge on 40 units",
              "severity": "high"}
    
    Anomaly types: price_mismatch, quantity_mismatch, missing_document,
                   date_inconsistency, total_mismatch, multiple_errors
    """
    global _anomaly_log
    
    try:
        data = json.loads(anomaly_input)
        
        # Add timestamp and status
        data['timestamp'] = datetime.now().isoformat()
        data['status'] = 'pending_human_review'
        data['anomaly_id'] = f"ANOM-{len(_anomaly_log) + 1:04d}"
        
        _anomaly_log.append(data)
        
        # Save to disk
        VECTOR_STORE_DIR.mkdir(exist_ok=True)
        with open(ANOMALY_LOG_PATH, 'w') as f:
            json.dump(_anomaly_log, f, indent=2)
        
        anomaly_type = data.get('anomaly_type', 'unknown')
        po_ref = data.get('po_ref', 'unknown')
        impact = data.get('impact', 'not specified')
        severity = data.get('severity', 'unknown')
        anomaly_id = data['anomaly_id']
        
        return (
            f"ANOMALY FLAGGED: {anomaly_id}\n"
            f"Type: {anomaly_type}\n"
            f"PO Reference: {po_ref}\n"
            f"Impact: {impact}\n"
            f"Severity: {severity}\n"
            f"Status: Pending human review\n"
            f"Logged to: {ANOMALY_LOG_PATH}\n"
            f"Human reviewer must confirm before inclusion in final report."
        )
    
    except json.JSONDecodeError:
        return "Invalid JSON input for anomaly flagging."
    except Exception as e:
        return f"Error flagging anomaly: {str(e)}"


# ============================================================
# AGENT SETUP
# LangChain ReAct agent with memory
# ============================================================

REACT_PROMPT = PromptTemplate.from_template("""
You are an expert supply chain document auditor. Your role is to analyse 
supply chain document sets — Purchase Orders (PO), Invoices (INV), and 
Delivery Notes (DN) — and detect any anomalies or inconsistencies between them.

You have access to the following tools:
{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: think about what you need to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: your complete anomaly report

IMPORTANT INSTRUCTIONS:
1. Always start by checking all three documents exist (PO, Invoice, DN)
2. Search for PO first, then Invoice, then Delivery Note
3. Compare unit prices, quantities, and totals across documents
4. Flag ANY discrepancy using flag_anomaly tool
5. If a document is missing, flag it as missing_document anomaly
6. Always provide a clear final report stating: CONSISTENT or ANOMALY DETECTED

Previous conversation:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}
""")


def create_agent():
    """
    Create and return the LangChain ReAct agent with all five tools.
    
    Uses GPT-4o-mini for cost efficiency while maintaining strong
    reasoning capability for document comparison tasks.
    Memory layer follows Liu et al. (2024) MATRIX approach —
    maintaining context across multi-turn verification queries.
    """
    # Load vector store on agent creation
    get_resources()
    
    # Define tools list
    tools = [
        search_documents,
        compare_values,
        calculate_total,
        check_document_exists,
        flag_anomaly,
    ]
    
    # Initialise LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,  # deterministic for auditing
        api_key=os.getenv('OPENAI_API_KEY')
    )
    
    # Memory layer — maintains context across queries in a session
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=False
    )
    
    # Create ReAct agent
    agent = create_react_agent(llm, tools, REACT_PROMPT)
    
    # Create agent executor with HITL-aware settings
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,           # show ReAct reasoning trace
        max_iterations=15,      # prevent infinite loops
        early_stopping_method="force",
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )
    
    return agent_executor


# ============================================================
# HITL HANDLER
# Mid-retrieval intervention point — Lazaros et al. (2026)
# ============================================================

def check_hitl_required(po_ref: str) -> bool:
    """
    Check if any document in a transaction set requires HITL review
    before the agent proceeds with comparison.
    
    This implements the mid-retrieval HITL intervention point:
    if extraction confidence is low on any document, human review
    is requested before the agent compares values.
    """
    _, _, document_results = get_resources()
    
    low_confidence_docs = [
        dr for dr in document_results
        if dr.po_ref == po_ref and dr.hitl_required
    ]
    
    if low_confidence_docs:
        print(f"\n{'='*60}")
        print("HITL INTERVENTION REQUIRED — Mid-retrieval check")
        print(f"{'='*60}")
        print(f"Low confidence extraction detected for {po_ref}:")
        for doc in low_confidence_docs:
            print(f"  - {Path(doc.source_file).name}: "
                  f"confidence {doc.confidence_score:.2f}")
            for warning in doc.extraction_warnings:
                print(f"    Warning: {warning}")
        print("\nHuman review recommended before agent proceeds.")
        print(f"{'='*60}\n")
        return True
    
    return False


# ============================================================
# ANOMALY REPORT GENERATOR
# ============================================================

def generate_anomaly_report() -> str:
    """
    Generate a structured anomaly report from all flagged anomalies.
    This is the final output of the system for human review.
    """
    global _anomaly_log
    
    if not _anomaly_log:
        # Try loading from disk
        if ANOMALY_LOG_PATH.exists():
            with open(ANOMALY_LOG_PATH, 'r') as f:
                _anomaly_log = json.load(f)
    
    if not _anomaly_log:
        return "No anomalies detected in this session."
    
    report_lines = [
        "=" * 60,
        "SUPPLY CHAIN DOCUMENT ANOMALY REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total anomalies flagged: {len(_anomaly_log)}",
        "=" * 60,
    ]
    
    for anomaly in _anomaly_log:
        report_lines.extend([
            f"\nAnomaly ID: {anomaly.get('anomaly_id', 'N/A')}",
            f"Type: {anomaly.get('anomaly_type', 'N/A')}",
            f"PO Reference: {anomaly.get('po_ref', 'N/A')}",
            f"Documents: {anomaly.get('document1', 'N/A')} vs "
            f"{anomaly.get('document2', 'N/A')}",
            f"Field: {anomaly.get('field', 'N/A')}",
            f"Values: {anomaly.get('value1', 'N/A')} vs "
            f"{anomaly.get('value2', 'N/A')}",
            f"Impact: {anomaly.get('impact', 'N/A')}",
            f"Severity: {anomaly.get('severity', 'N/A')}",
            f"Status: {anomaly.get('status', 'N/A')}",
            f"Timestamp: {anomaly.get('timestamp', 'N/A')}",
            "-" * 40,
        ])
    
    return "\n".join(report_lines)


# ============================================================
# MAIN — TEST THE AGENT
# ============================================================

if __name__ == "__main__":
    print("Initialising Supply Chain Document Intelligence Agent...")
    print("=" * 60)
    
    # Create agent
    agent = create_agent()
    
    print("\nAgent ready. Running test verification on Set 06 (price mismatch)...")
    print("=" * 60)
    
    # Test query — Set 06 has a known price mismatch
    po_ref = "PO-2026-2001"
    
    # Check HITL before proceeding
    hitl_needed = check_hitl_required(po_ref)
    if hitl_needed:
        response = input("\nProceed with agent verification despite low confidence? (yes/no): ")
        if response.lower() != 'yes':
            print("Agent verification paused for human review.")
            exit()
    
    # Run agent
    query = (
        f"Please verify the supply chain documents for purchase order {po_ref}. "
        f"Check whether the invoice matches the purchase order in terms of "
        f"unit prices, quantities, and total amounts. "
        f"Flag any discrepancies you find."
    )
    
    print(f"\nQuery: {query}\n")
    print("=" * 60)
    
    result = agent.invoke({"input": query})
    
    print("\n" + "=" * 60)
    print("AGENT FINAL ANSWER:")
    print("=" * 60)
    print(result.get('output', 'No output generated'))
    
    # Print anomaly report
    print("\n" + "=" * 60)
    print(generate_anomaly_report())
