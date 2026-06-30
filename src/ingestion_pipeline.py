"""
Document Ingestion & Structure-Aware Chunking Module
Agentic RAG System for Automated Document Intelligence in Supply Chain Operations

This module extracts text from PDF documents and splits it into
structure-aware chunks (header / line-items / summary) rather than
naive fixed-length splitting. This is the core technical contribution
identified in the dissertation proposal.
"""

import pdfplumber
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DocumentChunk:
    """A single structure-aware chunk of a document."""
    chunk_type: str          # 'header', 'line_item', 'summary'
    content: str             # the actual text content
    source_file: str         # which PDF this came from
    metadata: dict = field(default_factory=dict)  # extra extracted fields


def extract_text_from_pdf(filepath: str) -> str:
    """Extract raw text from a PDF file using pdfplumber."""
    full_text = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
    return "\n".join(full_text)


def classify_line(line: str) -> str:
    """
    Classify a single line of extracted text as header, line_item,
    or summary, based on pattern matching rather than fixed position.
    This is necessary because our 5 supplier formats place these
    elements in different positions within the document.
    """
    line_lower = line.lower().strip()

    # Table column header row (e.g. "Line Part No. Description Qty Unit £")
    # Must be checked BEFORE summary, since it can contain words like
    # "Total" as a column name without being an actual total line.
    column_header_keywords = ['part no', 'description', 'qty', 'desc|qty',
                               'item_code']
    has_column_keywords = sum(1 for kw in column_header_keywords if kw in line_lower) >= 2
    has_actual_number = bool(re.search(r'\d+\.\d{2}', line))
    if has_column_keywords and not has_actual_number:
        return 'header'

    # Summary indicators: totals, VAT, subtotal, grand total, amount due
    # Require an actual monetary/numeric value to be present, otherwise
    # a column header containing the word "Total" gets misclassified.
    summary_keywords = ['subtotal', 'vat', 'grand total', 'amount due',
                         'total due', 'total payable', 'total_due',
                         'vat_20pct', 'total|']
    if any(kw in line_lower for kw in summary_keywords) and has_actual_number:
        return 'summary'
    if 'total:' in line_lower and has_actual_number:
        return 'summary'

    # Header indicators: PO/Invoice/DN numbers, dates, company names,
    # "bill to", "ship to", "date issued"
    header_keywords = ['po ref', 'po number', 'po_number', 'invoice no',
                        'invoice_no', 'dn no', 'dn_no', 'bill to',
                        'ship to', 'delivered to', 'date:', 'po #',
                        'order #', 'order confirmation', 'del. note',
                        'invoice date', 'delivery date', 'date delivered',
                        'date issued', 'issued:', 'received_by',
                        'delivered_to', 'customer:', 'customer po']
    if any(kw in line_lower for kw in header_keywords):
        return 'header'

    # Line item indicators: contains a product code pattern (letters-digits)
    # and a quantity/price pattern, OR contains '|' pipe delimiters with
    # an item code, OR matches "qty @ price" style prose
    has_product_code = bool(re.search(r'\b[A-Z]{2,3}-\d{3}\b', line))
    has_price = bool(re.search(r'£\s?\d+[.,]\d{2}', line) or
                      re.search(r'\d+\.\d{2}', line))
    has_qty_pattern = bool(re.search(r'\bx?\d+\s*(units?|@)\b', line_lower))

    if has_product_code and (has_price or has_qty_pattern):
        return 'line_item'

    # Pipe-delimited line items (Vantage style export)
    if '|' in line and re.search(r'[A-Z]{2}-\d{3}', line):
        return 'line_item'

    # Prose-style line items mentioning a product code, even without
    # a price on the same line (price may be on the next line) —
    # catches Northgate/Coastal style multi-line descriptions
    if has_product_code:
        return 'line_item'

    return 'unclassified'


def chunk_document(filepath: str) -> List[DocumentChunk]:
    """
    Extract text from a PDF and split it into structure-aware chunks.
    Groups consecutive lines of the same classification together,
    rather than splitting every line into its own chunk.
    """
    raw_text = extract_text_from_pdf(filepath)
    lines = [l for l in raw_text.split('\n') if l.strip()]

    chunks = []
    current_type = None
    current_lines = []

    for line in lines:
        line_type = classify_line(line)

        # Treat 'unclassified' lines as continuations of whatever
        # section we're currently in (e.g. prose continuing onto a
        # second line, or address lines following a header)
        effective_type = line_type if line_type != 'unclassified' else current_type

        if effective_type != current_type and current_lines:
            # Flush the current group as a chunk
            chunks.append(DocumentChunk(
                chunk_type=current_type or 'unclassified',
                content='\n'.join(current_lines),
                source_file=filepath
            ))
            current_lines = []

        current_type = effective_type
        current_lines.append(line)

    # Flush final group
    if current_lines:
        chunks.append(DocumentChunk(
            chunk_type=current_type or 'unclassified',
            content='\n'.join(current_lines),
            source_file=filepath
        ))

    return chunks


def extract_key_fields(chunks: List[DocumentChunk]) -> dict:
    """
    Extract structured fields (PO number, total amount, dates, etc.)
    from the chunked document. This is what the agent's compare_values()
    and calculate_total() tools will actually use for reasoning.
    """
    full_text = '\n'.join(c.content for c in chunks)

    fields = {}

    # PO reference — match formats like "PO-2026-1001" directly, rather
    # than relying on the label text before it (which varies too much
    # across formats and was previously grabbing the wrong following word)
    po_match = re.search(r'\bPO-\d{4}-\d{3,4}\b', full_text)
    if po_match:
        fields['po_ref'] = po_match.group(0)
    else:
        # Fallback for informal references like "your PO 1005" where the
        # full PO-2026-XXXX format isn't repeated on this document —
        # common on small-supplier style invoices (e.g. Coastal Fleet)
        informal_match = re.search(r'(?:your )?PO[\s#]*(\d{3,4})\b', full_text, re.IGNORECASE)
        if informal_match:
            fields['po_ref'] = f"PO-(informal ref: {informal_match.group(1)})"

    # Invoice reference — match formats like "INV-MV-7741" or "INV-VE-8804"
    # or the abbreviated "CFS-2207" style used by Coastal Fleet
    inv_match = re.search(r'\bINV-[A-Z]{2,3}-\d{3,4}\b', full_text)
    if not inv_match:
        inv_match = re.search(r'\b(?:inv no\.?\s*)?([A-Z]{2,4}-\d{3,4})\b(?=.*your PO|.*\(your)',
                               full_text, re.IGNORECASE)
    if inv_match:
        fields['invoice_ref'] = inv_match.group(0) if inv_match.lastindex is None else inv_match.group(1)

    # Delivery note reference
    dn_match = re.search(r'\bDN-\d{3,4}\b', full_text)
    if not dn_match:
        dn_match = re.search(r'del\.?\s*note\s*(\d{3,4})', full_text, re.IGNORECASE)
    if dn_match:
        fields['dn_ref'] = dn_match.group(0)

    # Total amount — search in priority order, since a document may
    # contain multiple monetary figures (subtotal, VAT, total). We want
    # the FINAL/GRAND total specifically, with subtotal only as fallback.
    total_priority_patterns = [
        r'GRAND TOTAL:?\s*£?\s?([\d,]+(?:\.\d{2})?)',
        r'AMOUNT DUE:?\s*£?\s?([\d,]+(?:\.\d{2})?)',
        r'Total amount now due:?\s*£([\d,]+(?:\.\d{2})?)',
        r'TOTAL DUE:?\s*£?\s?([\d,]+(?:\.\d{2})?)',
        r'TOTAL PAYABLE\s*£?\s?([\d,]+(?:\.\d{2})?)',
        r'TOTAL_DUE\|([\d,]+(?:\.\d{2})?)',
        r'=\s*TOTAL DUE:\s*£([\d,]+(?:\.\d{2})?)',
        r'TOTAL:\s*£([\d,]+(?:\.\d{2})?)',
        r'\bTOTAL\|([\d,]+(?:\.\d{2})?)',
        r'Order Total:\s*£([\d,]+(?:\.\d{2})?)',
        r'bringing the (?:order )?total to £([\d,]+(?:\.\d{2})?)',
        # Fallback: subtotal, only used if nothing above matched
        r'Subtotal:?\s*£?\s?([\d,]+(?:\.\d{2})?)',
        r'subtotal\s*£([\d,]+(?:\.\d{2})?)',
        r'SUBTOTAL\|([\d,]+(?:\.\d{2})?)',
        r'totalling\s*£([\d,]+(?:\.\d{2})?)',
    ]
    for pattern in total_priority_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            fields['total_amount'] = float(match.group(1).replace(',', ''))
            break

    # Product codes and quantities — deduplicated, one entry per
    # distinct product code found in line_item chunks
    line_item_chunks = [c for c in chunks if c.chunk_type == 'line_item']
    seen_codes = set()
    items = []
    for chunk in line_item_chunks:
        codes = re.findall(r'\b([A-Z]{2,3}-\d{3})\b', chunk.content)
        for code in codes:
            if code not in seen_codes:
                seen_codes.add(code)
                items.append({'code': code, 'raw_text': chunk.content})
    fields['items'] = items

    return fields


# ============================================================
# TEST / DEMONSTRATION
# ============================================================
if __name__ == "__main__":
    test_files = [
        "data/document_sets/set01_meridian/PO-2026-1001.pdf",
        "data/document_sets/set01_meridian/INV-MV-7741.pdf",
        "data/document_sets/set01_meridian/DN-9921.pdf",
        "data/document_sets/set03_northgate/PO-2026-1003.pdf",
        "data/document_sets/set03_northgate/INV-NT-5512.pdf",
        "data/document_sets/set04_vantage/PO-2026-1004.pdf",
        "data/document_sets/set04_vantage/INV-VE-8804.pdf",
        "data/document_sets/set05_coastal/PO-2026-1005.pdf",
        "data/document_sets/set05_coastal/INV-CFS-2207.pdf",
        "data/document_sets/discrepancy_sets/set06_meridian_price_mismatch/PO-2026-2001.pdf",
        "data/document_sets/discrepancy_sets/set06_meridian_price_mismatch/INV-MV-8102.pdf",
        "data/document_sets/discrepancy_sets/set09_vantage_quantity_mismatch/PO-2026-2004.pdf",
        "data/document_sets/discrepancy_sets/set09_vantage_quantity_mismatch/DN-9215.pdf",
    ]

    for filepath in test_files:
        print(f"\n{'='*70}")
        print(f"DOCUMENT: {filepath}")
        print('='*70)

        chunks = chunk_document(filepath)
        fields = extract_key_fields(chunks)
        print(f"  PO Ref:        {fields.get('po_ref', 'NOT FOUND')}")
        print(f"  Invoice Ref:   {fields.get('invoice_ref', 'NOT FOUND')}")
        print(f"  DN Ref:        {fields.get('dn_ref', 'NOT FOUND')}")
        print(f"  Total Amount:  £{fields.get('total_amount', 'NOT FOUND')}")
        print(f"  Items found:   {[item['code'] for item in fields.get('items', [])]}")
