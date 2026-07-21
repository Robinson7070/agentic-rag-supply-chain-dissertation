"""
Document Ingestion & Structure-Aware Chunking Module — Version 2
Agentic RAG System for Automated Document Intelligence in Supply Chain Operations

UPGRADES FROM V1 (informed by literature review):
- Document type detection (PO / Invoice / DN) from content — Hamri et al. (2024)
- Supplier format detection (Meridian / Castlegate / Northgate / Vantage / Coastal)
- Enriched chunk metadata to prevent semantic entanglement — Loghmani (2026)
- Structured metadata tagging on every chunk — Cheerla (2025), Shah et al. (2026)
- Line item quantity and unit price extraction — Cesista et al. (2024)
- Document-level confidence scoring for HITL trigger — Adedokun et al. (2026)
- Consistent structured output schema across all 5 supplier formats
"""

import pdfplumber
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class DocumentChunk:
    """
    A single structure-aware chunk of a supply chain document.
    
    Enriched with metadata following Cheerla (2025) and Shah et al. (2026)
    recommendations for structured metadata tagging before embedding.
    Metadata tags prevent semantic entanglement (Loghmani, 2026) by
    ensuring each chunk's embedding accurately represents its specific
    content type rather than a confused mixture of document sections.
    """
    chunk_type: str           # 'header', 'line_item', 'summary', 'unclassified'
    content: str              # the actual text content
    source_file: str          # which PDF this came from
    chunk_index: int          # position of this chunk within the document
    document_type: str        # 'purchase_order', 'invoice', 'delivery_note', 'unknown'
    supplier_format: str      # 'meridian', 'castlegate', 'northgate', 'vantage', 'coastal', 'unknown'
    po_ref: str               # PO reference extracted from this document
    invoice_ref: str          # Invoice reference extracted from this document
    dn_ref: str               # Delivery note reference extracted from this document
    confidence: float         # extraction confidence score 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentResult:
    """
    Complete structured result for a single processed document.
    
    Consistent output schema regardless of supplier format,
    following Cesista et al. (2024) structured output approach.
    Includes confidence scoring to trigger HITL review when needed,
    following Adedokun et al. (2026) and Lazaros et al. (2026).
    """
    source_file: str
    document_type: str           # 'purchase_order', 'invoice', 'delivery_note', 'unknown'
    supplier_format: str         # which of the 5 formats this document uses
    po_ref: str                  # PO reference number or 'NOT_FOUND'
    invoice_ref: str             # Invoice reference or 'NOT_FOUND'
    dn_ref: str                  # Delivery note reference or 'NOT_FOUND'
    total_amount: Optional[float] # Total amount or None
    items: List[Dict]            # list of line items with code, qty, unit_price
    chunks: List[DocumentChunk]  # all structure-aware chunks
    confidence_score: float      # overall extraction confidence 0.0 to 1.0
    hitl_required: bool          # True if confidence below threshold
    extraction_warnings: List[str] # list of any extraction issues flagged


# ============================================================
# DOCUMENT TYPE DETECTION
# Hamri et al. (2024) — detecting document anatomy from content
# ============================================================

def detect_document_type(text: str) -> str:
    """
    Detect whether a document is a Purchase Order, Invoice, or Delivery Note
    by scanning for type-specific indicators in the text content.
    
    This content-based detection is more reliable than filename-based
    detection since real-world documents may have inconsistent naming
    conventions across suppliers.
    """
    text_lower = text.lower()

    # Purchase Order indicators
    po_indicators = [
        'purchase order', 'order confirmation', 'po number', 'po ref',
        'po #', 'po_number', 'order no', 'we hereby place', 'please supply',
        'order the following', 'authorised to purchase'
    ]
    po_score = sum(1 for kw in po_indicators if kw in text_lower)

    # Invoice indicators
    invoice_indicators = [
        'invoice', 'inv no', 'inv_no', 'invoice number', 'invoice date',
        'amount due', 'payment due', 'remittance', 'vat number',
        'bill to', 'please remit', 'payment terms'
    ]
    invoice_score = sum(1 for kw in invoice_indicators if kw in text_lower)

    # Delivery Note indicators
    dn_indicators = [
        'delivery note', 'del. note', 'dn no', 'dn_no', 'delivery confirmation',
        'goods delivered', 'received by', 'delivered to', 'delivery date',
        'despatch note', 'dispatch note', 'goods receipt'
    ]
    dn_score = sum(1 for kw in dn_indicators if kw in text_lower)

    # Return the type with the highest score
    # Apply tiebreaker: if PO score is within 1 of invoice score,
    # prefer purchase_order since PO documents often mention invoices
    # in payment terms sections, causing false invoice classification
    scores = {
        'purchase_order': po_score,
        'invoice': invoice_score,
        'delivery_note': dn_score
    }
    
    # Strong PO override: if document has explicit PO number pattern
    # and PO indicators, classify as PO regardless of invoice mentions
    has_po_number = bool(re.search(r'Purchase Order No\.?|PO_NUMBER|PURCHASE ORDER', text, re.IGNORECASE))
    has_dn_markers = bool(re.search(r'delivery note|del\. note|DN_NO|DELIVERY NOTE', text, re.IGNORECASE))
    
    if has_dn_markers and dn_score > 0:
        return 'delivery_note'
    
    # Strong invoice override: explicit INVOICE heading takes priority
    has_invoice_heading = bool(re.search(
        r'^INVOICE$|^TAX INVOICE$|INVOICE\s*\n|INVOICE EXPORT|INVOICE_NO\|',
        text, re.IGNORECASE | re.MULTILINE
    ))
    if has_invoice_heading and invoice_score > 0:
        return 'invoice'
    
    if has_po_number and po_score > 0:
        return 'purchase_order'
    
    best_type = max(scores, key=scores.get)
    if scores[best_type] == 0:
        return 'unknown'
    return best_type


# ============================================================
# SUPPLIER FORMAT DETECTION
# ============================================================

def detect_supplier_format(text: str) -> str:
    """
    Detect which of the 5 supplier formats this document uses.
    Each format has distinctive structural markers.
    """
    # Vantage — pipe-delimited format
    if '|' in text and re.search(r'[A-Z]{2}-\d{3}\|', text):
        return 'vantage'

    # Northgate — prose/letter format with company name or characteristic phrases
    if re.search(r'(northgate|please supply the following|'
                 r'we have supplied|units of.*product code|'
                 r'authorised on behalf)', text, re.IGNORECASE):
        return 'northgate'

    # Coastal Fleet — informal small-supplier format
    if re.search(r'(coastal fleet|cfs-\d{4}|as per your po|'
                 r'payment within \d+ days)', text, re.IGNORECASE):
        return 'coastal'

    # Castlegate — modern minimal format
    if re.search(r'(castlegate|cgl-|invoice #\s*[A-Z])', text, re.IGNORECASE):
        return 'castlegate'

    # Meridian — formal corporate with tables (default for structured docs)
    if re.search(r'(meridian|mv-\d{4}|vehicle parts|'
                 r'registered in england)', text, re.IGNORECASE):
        return 'meridian'

    return 'unknown'


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(filepath: str) -> str:
    """Extract raw text from a PDF file using pdfplumber."""
    full_text = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
    return "\n".join(full_text)


# ============================================================
# LINE CLASSIFICATION
# Structure-aware chunking — core novel contribution
# Validated by Cheerla (2025), Hamri et al. (2024), Loghmani (2026)
# ============================================================

def classify_line(line: str) -> str:
    """
    Classify a single line as header, line_item, or summary.
    
    Pattern-based semantic classification rather than positional splitting —
    this is the core of the structure-aware chunking contribution that
    addresses the layout variability problem identified by Saout et al. (2024)
    and the semantic entanglement problem formalised by Loghmani (2026).
    """
    line_lower = line.lower().strip()
    has_actual_number = bool(re.search(r'\d+\.\d{2}', line))

    # --- SUMMARY classification ---
    summary_keywords = [
        'subtotal', 'vat', 'grand total', 'amount due',
        'total due', 'total payable', 'total_due',
        'vat_20pct', 'total|', 'amount now due'
    ]
    if any(kw in line_lower for kw in summary_keywords) and has_actual_number:
        return 'summary'
    if 'total:' in line_lower and has_actual_number:
        return 'summary'
    # Pipe-delimited summary totals (Vantage format)
    if re.search(r'TOTAL\|[\d.]+', line):
        return 'summary'

    # --- HEADER classification ---
    # Table column header rows
    column_header_keywords = ['part no', 'description', 'qty', 'desc|qty', 'item_code']
    has_column_keywords = sum(1 for kw in column_header_keywords if kw in line_lower) >= 2
    if has_column_keywords and not has_actual_number:
        return 'header'

    # Document header fields
    header_keywords = [
        'po ref', 'po number', 'po_number', 'invoice no', 'invoice_no',
        'dn no', 'dn_no', 'bill to', 'ship to', 'delivered to',
        'date:', 'po #', 'order #', 'order confirmation', 'del. note',
        'invoice date', 'delivery date', 'date delivered', 'date issued',
        'issued:', 'received_by', 'delivered_to', 'customer:', 'customer po',
        'purchase order', 'delivery note', 'invoice number', 'supplier:',
        'vendor:', 'from:', 'to:', 'payment terms', 'vat no'
    ]
    if any(kw in line_lower for kw in header_keywords):
        return 'header'

    # --- LINE ITEM classification ---
    has_product_code = bool(re.search(r'\b[A-Z]{2,3}-\d{3}\b', line))
    has_price = bool(
        re.search(r'£\s?\d+[.,]\d{2}', line) or
        re.search(r'\d+\.\d{2}', line)
    )
    has_qty_pattern = bool(re.search(r'\bx?\d+\s*(units?|@)\b', line_lower))

    if has_product_code and (has_price or has_qty_pattern):
        return 'line_item'

    # Pipe-delimited line items (Vantage format)
    if '|' in line and re.search(r'[A-Z]{2}-\d{3}', line):
        return 'line_item'

    # Product code present even without price on same line
    # (Northgate/Coastal multi-line descriptions)
    if has_product_code:
        return 'line_item'

    return 'unclassified'


# ============================================================
# DOCUMENT CHUNKING
# ============================================================

def chunk_document(filepath: str, document_type: str,
                   supplier_format: str, po_ref: str,
                   invoice_ref: str, dn_ref: str) -> List[DocumentChunk]:
    """
    Extract text from a PDF and split into structure-aware chunks.
    
    Each chunk is enriched with document-level metadata following
    Cheerla (2025) metadata-aware retrieval recommendations and
    Shah et al. (2026) structured metadata enrichment approach.
    This enables the vector store to filter by document type and
    reference number before semantic similarity matching.
    """
    raw_text = extract_text_from_pdf(filepath)
    lines = [l for l in raw_text.split('\n') if l.strip()]

    chunks = []
    current_type = None
    current_lines = []
    chunk_index = 0

    for line in lines:
        line_type = classify_line(line)
        effective_type = line_type if line_type != 'unclassified' else current_type

        if effective_type != current_type and current_lines:
            # Calculate per-chunk confidence based on content richness
            chunk_content = '\n'.join(current_lines)
            chunk_confidence = _calculate_chunk_confidence(
                chunk_content, current_type or 'unclassified'
            )

            chunks.append(DocumentChunk(
                chunk_type=current_type or 'unclassified',
                content=chunk_content,
                source_file=filepath,
                chunk_index=chunk_index,
                document_type=document_type,
                supplier_format=supplier_format,
                po_ref=po_ref,
                invoice_ref=invoice_ref,
                dn_ref=dn_ref,
                confidence=chunk_confidence,
                metadata={
                    'char_count': len(chunk_content),
                    'line_count': len(current_lines),
                }
            ))
            chunk_index += 1
            current_lines = []

        current_type = effective_type
        current_lines.append(line)

    # Flush final group
    if current_lines:
        chunk_content = '\n'.join(current_lines)
        chunks.append(DocumentChunk(
            chunk_type=current_type or 'unclassified',
            content=chunk_content,
            source_file=filepath,
            chunk_index=chunk_index,
            document_type=document_type,
            supplier_format=supplier_format,
            po_ref=po_ref,
            invoice_ref=invoice_ref,
            dn_ref=dn_ref,
            confidence=_calculate_chunk_confidence(
                chunk_content, current_type or 'unclassified'
            ),
            metadata={
                'char_count': len(chunk_content),
                'line_count': len(current_lines),
            }
        ))

    return chunks


def _calculate_chunk_confidence(content: str, chunk_type: str) -> float:
    """
    Calculate a confidence score for a chunk based on how much
    structured information was successfully extracted from it.
    
    Used to trigger HITL review for low-confidence extractions,
    following the exception-handling intervention design validated
    by Lazaros et al. (2026) and Adedokun et al. (2026).
    """
    score = 0.5  # base score

    if chunk_type == 'line_item':
        has_code = bool(re.search(r'\b[A-Z]{2,3}-\d{3}\b', content))
        has_price = bool(re.search(r'£\s?\d+|\d+\.\d{2}', content))
        has_qty = bool(re.search(r'\d+\s*(units?|x\d)', content, re.IGNORECASE))
        score += 0.2 if has_code else 0
        score += 0.2 if has_price else 0
        score += 0.1 if has_qty else 0

    elif chunk_type == 'header':
        has_ref = bool(re.search(r'(PO|INV|DN)-[\w-]+', content))
        has_date = bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', content))
        score += 0.3 if has_ref else 0
        score += 0.2 if has_date else 0

    elif chunk_type == 'summary':
        has_amount = bool(re.search(r'£\s?[\d,]+\.\d{2}', content))
        score += 0.5 if has_amount else 0

    return min(round(score, 2), 1.0)


# ============================================================
# KEY FIELD EXTRACTION
# Enhanced with qty/price per line item — Cesista et al. (2024)
# ============================================================

def extract_key_fields(full_text: str) -> dict:
    """
    Extract structured fields from document text.
    
    Enhanced from V1 to include unit price and quantity per line item,
    following Cesista et al. (2024) line item recognition approach.
    This is essential for the agent's compare_values() tool to detect
    price and quantity discrepancies across linked documents.
    """
    fields = {}

    # --- PO Reference ---
    po_match = re.search(r'\bPO-\d{4}-\d{3,4}\b', full_text)
    if po_match:
        fields['po_ref'] = po_match.group(0)
    else:
        informal_match = re.search(
            r'(?:your )?PO[\s#]*(\d{3,4})\b', full_text, re.IGNORECASE
        )
        if informal_match:
            fields['po_ref'] = f"PO-(informal ref: {informal_match.group(1)})"
        else:
            fields['po_ref'] = 'NOT_FOUND'

    # --- Invoice Reference ---
    inv_match = re.search(r'\bINV-[A-Z]{2,3}-\d{3,4}\b', full_text)
    if not inv_match:
        inv_match = re.search(
            r'\b(?:inv no\.?\s*)?([A-Z]{2,4}-\d{3,4})\b(?=.*your PO|.*\(your)',
            full_text, re.IGNORECASE
        )
    fields['invoice_ref'] = (
        inv_match.group(0) if inv_match and inv_match.lastindex is None
        else inv_match.group(1) if inv_match else 'NOT_FOUND'
    )

    # --- Delivery Note Reference ---
    dn_match = re.search(r'\bDN-\d{3,4}\b', full_text)
    if not dn_match:
        dn_match = re.search(
            r'del\.?\s*note\s*(\d{3,4})', full_text, re.IGNORECASE
        )
    fields['dn_ref'] = dn_match.group(0) if dn_match else 'NOT_FOUND'

    # --- Total Amount ---
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
        r'Subtotal:?\s*£?\s?([\d,]+(?:\.\d{2})?)',
        r'subtotal\s*£([\d,]+(?:\.\d{2})?)',
        r'SUBTOTAL\|([\d,]+(?:\.\d{2})?)',
        r'totalling\s*£([\d,]+(?:\.\d{2})?)',
    ]
    fields['total_amount'] = None
    for pattern in total_priority_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            fields['total_amount'] = float(match.group(1).replace(',', ''))
            break

    # --- Line Items with Quantity and Unit Price ---
    # Enhanced from V1 — now extracts qty and unit_price per item
    # following Cesista et al. (2024) line item recognition approach
    # 
    # IMPORTANT: Pipe pattern (Vantage) runs FIRST because structured
    # pattern can accidentally match partial pipe-delimited numbers
    items = []
    seen_codes = set()

    # Pattern 1: Vantage pipe-delimited FIRST — most precise format
    # "EC-902|Vehicle Battery 12V 90Ah|18|95.00|1710.00"
    # Must run before structured pattern to avoid partial number matches
    pipe_pattern = re.finditer(
        r'([A-Z]{2,3}-\d{3})\|[^|\n]+\|(\d+)\|([\d]+\.\d{2})\|([\d]+\.\d{2})',
        full_text
    )
    for match in pipe_pattern:
        code = match.group(1)
        if code not in seen_codes:
            seen_codes.add(code)
            try:
                items.append({
                    'code': code,
                    'quantity': int(match.group(2)),
                    'unit_price': float(match.group(3)),
                    'line_total': float(match.group(4)),
                    'raw_text': match.group(0)
                })
            except (ValueError, IndexError):
                pass

    # Pattern 2: Structured format — "VP-204 Brake Pads 40 units £42.00 £1,680.00"
    structured_pattern = re.finditer(
        r'\b([A-Z]{2,3}-\d{3})\b[^£\n]*?(\d+)\s*(?:units?|x)?\s*'
        r'(?:@\s*)?£?\s*([\d,]+\.\d{2})',
        full_text
    )
    for match in structured_pattern:
        code = match.group(1)
        if code not in seen_codes:
            seen_codes.add(code)
            try:
                qty = int(match.group(2))
                unit_price = float(match.group(3).replace(',', ''))
                items.append({
                    'code': code,
                    'quantity': qty,
                    'unit_price': unit_price,
                    'line_total': round(qty * unit_price, 2),
                    'raw_text': match.group(0)
                })
            except (ValueError, IndexError):
                items.append({
                    'code': code,
                    'quantity': None,
                    'unit_price': None,
                    'line_total': None,
                    'raw_text': match.group(0)
                })

    # Pattern 2a: Vantage DN delivery format
    # "EC-902|Vehicle Battery 12V 90Ah|15|GOOD" (QTY_DELIVERED, no price)
    dn_pipe_pattern = re.finditer(
        r'([A-Z]{2,3}-\d{3})\|[^|\n]+\|(\d+)\|(?:GOOD|DAMAGED|PARTIAL|SHORT)',
        full_text, re.IGNORECASE
    )
    for match in dn_pipe_pattern:
        code = match.group(1)
        if code not in seen_codes:
            seen_codes.add(code)
            try:
                items.append({
                    'code': code,
                    'quantity': int(match.group(2)),
                    'unit_price': None,
                    'line_total': None,
                    'raw_text': match.group(0),
                    'note': 'qty_delivered'
                })
            except (ValueError, IndexError):
                pass

    # Pattern 2b: Northgate prose format
    # "24 (twenty-four) units of ... (product code TY-440), at a unit price of £210.00"
    prose_pattern = re.finditer(
        r'(\d+)\s+(?:\([^)]+\)\s+)?units? of[^(]+\((?:product code\s+)?([A-Z]{2,3}-\d{3})\)[^£]*£([\d,]+\.\d{2})',
        full_text, re.IGNORECASE
    )
    for match in prose_pattern:
        code = match.group(2)
        if code not in seen_codes:
            seen_codes.add(code)
            try:
                qty = int(match.group(1))
                unit_price = float(match.group(3).replace(',', ''))
                items.append({
                    'code': code,
                    'quantity': qty,
                    'unit_price': unit_price,
                    'line_total': round(qty * unit_price, 2),
                    'raw_text': match.group(0)
                })
            except (ValueError, IndexError):
                pass

    # Pattern 3: Fallback — just product codes with no qty/price on same line
    fallback_codes = re.findall(r'\b([A-Z]{2,3}-\d{3})\b', full_text)
    for code in fallback_codes:
        if code not in seen_codes:
            seen_codes.add(code)
            items.append({
                'code': code,
                'quantity': None,
                'unit_price': None,
                'line_total': None,
                'raw_text': code
            })

    fields['items'] = items
    return fields


# ============================================================
# CONFIDENCE SCORING AT DOCUMENT LEVEL
# ============================================================

def calculate_document_confidence(fields: dict, chunks: List[DocumentChunk]) -> float:
    """
    Calculate overall document extraction confidence.
    
    Based on how many key fields were successfully extracted.
    Documents below the HITL threshold (0.6) are flagged for
    human review before the agent proceeds with comparison,
    implementing the mid-retrieval HITL intervention point.
    """
    score = 0.0
    total_weight = 0.0

    # Each field contributes weighted to overall confidence
    field_weights = {
        'po_ref': 0.25,
        'invoice_ref': 0.20,
        'dn_ref': 0.15,
        'total_amount': 0.25,
        'items': 0.15
    }

    for field_name, weight in field_weights.items():
        total_weight += weight
        value = fields.get(field_name)
        if field_name == 'items':
            if value and len(value) > 0:
                # Extra confidence if we got qty and price too
                has_details = any(
                    item.get('quantity') is not None for item in value
                )
                score += weight if has_details else weight * 0.5
        elif value and value != 'NOT_FOUND':
            score += weight

    # Factor in average chunk confidence
    if chunks:
        avg_chunk_conf = sum(c.confidence for c in chunks) / len(chunks)
        score = (score / total_weight) * 0.7 + avg_chunk_conf * 0.3
    else:
        score = score / total_weight if total_weight > 0 else 0.0

    return round(min(score, 1.0), 2)


# ============================================================
# MAIN PROCESSING FUNCTION
# ============================================================

HITL_CONFIDENCE_THRESHOLD = 0.6  # Documents below this trigger human review


def process_document(filepath: str) -> DocumentResult:
    """
    Full pipeline for a single document.
    
    Runs detection, chunking, extraction, and confidence scoring,
    returning a consistent DocumentResult regardless of supplier format.
    This is the function the embeddings layer will call for every PDF.
    """
    warnings = []

    # Step 1: Extract raw text
    raw_text = extract_text_from_pdf(filepath)

    if not raw_text.strip():
        return DocumentResult(
            source_file=filepath,
            document_type='unknown',
            supplier_format='unknown',
            po_ref='NOT_FOUND',
            invoice_ref='NOT_FOUND',
            dn_ref='NOT_FOUND',
            total_amount=None,
            items=[],
            chunks=[],
            confidence_score=0.0,
            hitl_required=True,
            extraction_warnings=['Empty document — no text extracted']
        )

    # Step 2: Detect document type and supplier format
    document_type = detect_document_type(raw_text)
    supplier_format = detect_supplier_format(raw_text)

    if document_type == 'unknown':
        warnings.append('Document type could not be determined from content')
    if supplier_format == 'unknown':
        warnings.append('Supplier format could not be determined')

    # Step 3: Extract key fields
    fields = extract_key_fields(raw_text)

    po_ref = fields.get('po_ref', 'NOT_FOUND')
    invoice_ref = fields.get('invoice_ref', 'NOT_FOUND')
    dn_ref = fields.get('dn_ref', 'NOT_FOUND')

    if po_ref == 'NOT_FOUND':
        warnings.append('PO reference not found')
    if document_type == 'invoice' and invoice_ref == 'NOT_FOUND':
        warnings.append('Invoice reference not found in invoice document')
    if fields.get('total_amount') is None:
        warnings.append('Total amount not found')
    if not fields.get('items'):
        warnings.append('No line items extracted')

    # Step 4: Chunk document with enriched metadata
    chunks = chunk_document(
        filepath=filepath,
        document_type=document_type,
        supplier_format=supplier_format,
        po_ref=po_ref,
        invoice_ref=invoice_ref,
        dn_ref=dn_ref
    )

    # Step 5: Calculate document-level confidence
    confidence_score = calculate_document_confidence(fields, chunks)
    hitl_required = confidence_score < HITL_CONFIDENCE_THRESHOLD

    if hitl_required:
        warnings.append(
            f'Low confidence score ({confidence_score:.2f}) — '
            f'human review recommended before agent comparison'
        )

    return DocumentResult(
        source_file=filepath,
        document_type=document_type,
        supplier_format=supplier_format,
        po_ref=po_ref,
        invoice_ref=invoice_ref,
        dn_ref=dn_ref,
        total_amount=fields.get('total_amount'),
        items=fields.get('items', []),
        chunks=chunks,
        confidence_score=confidence_score,
        hitl_required=hitl_required,
        extraction_warnings=warnings
    )


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

        result = process_document(filepath)

        print(f"  Document Type:    {result.document_type}")
        print(f"  Supplier Format:  {result.supplier_format}")
        print(f"  PO Ref:           {result.po_ref}")
        print(f"  Invoice Ref:      {result.invoice_ref}")
        print(f"  DN Ref:           {result.dn_ref}")
        print(f"  Total Amount:     £{result.total_amount}")
        print(f"  Chunks:           {len(result.chunks)} "
              f"({sum(1 for c in result.chunks if c.chunk_type == 'header')} header, "
              f"{sum(1 for c in result.chunks if c.chunk_type == 'line_item')} line_item, "
              f"{sum(1 for c in result.chunks if c.chunk_type == 'summary')} summary)")
        print(f"  Confidence:       {result.confidence_score:.2f}")
        print(f"  HITL Required:    {result.hitl_required}")

        if result.items:
            print(f"  Line Items:")
            for item in result.items:
                qty = item.get('quantity', 'N/A')
                price = f"£{item.get('unit_price', 'N/A')}"
                total = f"£{item.get('line_total', 'N/A')}"
                print(f"    {item['code']} — qty: {qty}, "
                      f"unit price: {price}, line total: {total}")

        if result.extraction_warnings:
            print(f"  Warnings:")
            for w in result.extraction_warnings:
                print(f"    ! {w}")
