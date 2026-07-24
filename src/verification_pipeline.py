"""
Structured Verification Pipeline — Version 1
Agentic RAG System for Automated Document Intelligence in Supply Chain Operations

This module implements a deterministic verification pipeline that orchestrates
the five agent tools in a structured sequence rather than relying solely on
LLM reasoning for workflow decisions.

Design rationale:
- Deterministic tool orchestration ensures reproducible, auditable results
- LLM reasoning is used for anomaly interpretation, not workflow control
- This hybrid approach combines the reliability of structured code with
  the semantic understanding of LLM-based comparison
- Directly implements the cross-document reasoning contribution of the dissertation
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

sys.path.insert(0, 'src')

from agent import (
    search_documents,
    compare_values,
    calculate_total,
    check_document_exists,
    flag_anomaly,
    generate_anomaly_report,
    get_resources,
    _anomaly_log
)


# ============================================================
# STRUCTURED VERIFICATION PIPELINE
# ============================================================

def verify_document_set(po_ref: str, verbose: bool = True) -> Dict:
    """
    Run structured verification for a complete supply chain document set.
    
    Orchestrates the five tools in a deterministic sequence:
    1. Check all three documents exist
    2. Retrieve PO line items and totals
    3. Retrieve Invoice line items and totals
    4. Retrieve DN quantities
    5. Compare PO vs Invoice (prices, quantities, totals)
    6. Compare PO vs DN (quantities)
    7. Flag any discrepancies
    8. Return structured verification result
    
    Returns a verification result dict with:
    - consistent: bool (True if no anomalies found)
    - anomalies: list of detected anomalies
    - documents_checked: list of documents verified
    - missing_documents: list of missing documents
    - verification_details: step-by-step comparison results
    """
    
    result = {
        'po_ref': po_ref,
        'timestamp': datetime.now().isoformat(),
        'consistent': True,
        'anomalies': [],
        'documents_checked': [],
        'missing_documents': [],
        'verification_details': [],
        'hitl_triggered': False
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"VERIFYING: {po_ref}")
        print('='*60)
    
    # --------------------------------------------------------
    # STEP 1: Check all three documents exist
    # --------------------------------------------------------
    
    doc_types = ['purchase_order', 'invoice', 'delivery_note']
    doc_status = {}
    
    for doc_type in doc_types:
        check_result = check_document_exists.invoke(f"{po_ref}|{doc_type}")
        exists = check_result.strip().startswith('FOUND')
        doc_status[doc_type] = exists
        
        if exists:
            result['documents_checked'].append(doc_type)
            if verbose:
                print(f"  ✓ {doc_type} found")
        else:
            result['missing_documents'].append(doc_type)
            result['consistent'] = False
            if verbose:
                print(f"  ✗ {doc_type} MISSING")
            
            # Flag missing document anomaly
            flag_result = flag_anomaly.invoke(json.dumps({
                'anomaly_type': 'missing_document',
                'po_ref': po_ref,
                'document1': 'N/A',
                'document2': 'N/A',
                'field': 'document_existence',
                'value1': 'expected',
                'value2': 'missing',
                'impact': f'{doc_type} not found for {po_ref}',
                'severity': 'high'
            }))
            result['anomalies'].append({
                'type': 'missing_document',
                'document': doc_type,
                'severity': 'high'
            })
    
    # If PO is missing we cannot proceed
    if not doc_status.get('purchase_order'):
        result['verification_details'].append(
            'Cannot proceed — Purchase Order not found'
        )
        return result
    
    # --------------------------------------------------------
    # STEP 2: Retrieve PO data
    # --------------------------------------------------------
    
    if verbose:
        print(f"\n  Retrieving PO data...")
    
    po_search = search_documents.invoke(
        f"{po_ref} purchase order line items quantities prices"
    )
    
    # Extract items from search results
    po_items = _extract_items_from_search(po_search, 'purchase_order')
    po_total = _extract_total_from_search(po_search, 'purchase_order')
    
    if verbose:
        print(f"  PO items: {[(i['code'], i.get('quantity'), i.get('unit_price')) for i in po_items]}")
        print(f"  PO total: £{po_total}")
    
    result['verification_details'].append({
        'step': 'PO retrieval',
        'po_ref': po_ref,
        'items': po_items,
        'total': po_total
    })
    
    # --------------------------------------------------------
    # STEP 3: Retrieve Invoice data and compare with PO
    # --------------------------------------------------------
    
    if doc_status.get('invoice'):
        if verbose:
            print(f"\n  Retrieving Invoice data...")
        
        # Search specifically for invoice document type
        inv_search = search_documents.invoke(
            f"{po_ref} invoice unit price line items amount due"
        )
        
        inv_items = _extract_items_from_search(inv_search, 'invoice')
        inv_total = _extract_total_from_search(inv_search, 'invoice')
        inv_ref = _extract_ref_from_search(inv_search, 'invoice_ref')
        
        # If invoice items not found via type filter, try fallback with invoice ref
        if not inv_items and inv_ref != 'NOT_FOUND':
            inv_search2 = search_documents.invoke(
                f"{inv_ref} invoice line items unit price"
            )
            inv_items = _extract_items_from_search(inv_search2, 'invoice')
            inv_total = _extract_total_from_search(inv_search2, 'invoice') or inv_total
        
        if verbose:
            print(f"  Invoice items: {[(i['code'], i.get('quantity'), i.get('unit_price')) for i in inv_items]}")
            print(f"  Invoice total: £{inv_total}")
        
        result['verification_details'].append({
            'step': 'Invoice retrieval',
            'invoice_ref': inv_ref,
            'items': inv_items,
            'total': inv_total
        })
        
        # Compare PO vs Invoice
        po_inv_anomalies = _compare_po_invoice(
            po_ref, inv_ref, po_items, inv_items,
            po_total, inv_total, verbose
        )
        result['anomalies'].extend(po_inv_anomalies)
        if po_inv_anomalies:
            result['consistent'] = False
    
    # --------------------------------------------------------
    # STEP 4: Retrieve DN data and compare quantities with PO
    # --------------------------------------------------------
    
    if doc_status.get('delivery_note'):
        if verbose:
            print(f"\n  Retrieving Delivery Note data...")
        
        dn_search = search_documents.invoke(
            f"{po_ref} delivery note received quantity delivered"
        )
        
        dn_items = _extract_items_from_search(dn_search, 'delivery_note')
        dn_ref = _extract_ref_from_search(dn_search, 'dn_ref')
        
        # Fallback: try searching for proof of delivery
        if not dn_items:
            dn_search2 = search_documents.invoke(
                f"{po_ref} proof of delivery items received condition"
            )
            dn_items = _extract_items_from_search(dn_search2, 'delivery_note')
            dn_ref = _extract_ref_from_search(dn_search2, 'dn_ref') or dn_ref
        
        if verbose:
            print(f"  DN items: {[(i['code'], i.get('quantity')) for i in dn_items]}")
        
        result['verification_details'].append({
            'step': 'DN retrieval',
            'dn_ref': dn_ref,
            'items': dn_items
        })
        
        # Compare PO vs DN quantities
        dn_anomalies = _compare_po_dn(
            po_ref, dn_ref, po_items, dn_items, verbose
        )
        result['anomalies'].extend(dn_anomalies)
        if dn_anomalies:
            result['consistent'] = False
    
    # --------------------------------------------------------
    # STEP 5: Check date consistency if available
    # --------------------------------------------------------
    
    # Date consistency is checked via the search results metadata
    # Future enhancement: extract dates and verify PO date < Invoice date < DN date
    
    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------
    
    if verbose:
        print(f"\n  {'─'*50}")
        if result['consistent']:
            print(f"  RESULT: CONSISTENT — No anomalies detected")
        else:
            print(f"  RESULT: ANOMALY DETECTED — {len(result['anomalies'])} issue(s) found")
            for a in result['anomalies']:
                print(f"    • {a['type']}: {a.get('details', '')}")
    
    return result


def _extract_items_from_search(search_output: str, target_doc_type: str) -> List[Dict]:
    """Extract line items from search results for a specific document type."""
    import re
    
    items = []
    seen_codes = set()
    
    # Split search output into individual result blocks
    # Each block starts with [Result N]
    result_blocks = re.split(r'\[Result \d+\]', search_output)
    
    # Process each block separately to avoid cross-block contamination
    for block in result_blocks:
        if not block.strip():
            continue
        
        # Check if this block matches the target document type
        doc_type_match = re.search(r'Document Type:\s*(\S+)', block)
        if not doc_type_match:
            continue
        
        block_doc_type = doc_type_match.group(1).strip().lower()
        
        # Normalise for comparison
        target_normalised = target_doc_type.lower().replace('_', '')
        block_normalised = block_doc_type.replace('_', '').replace(' ', '')
        
        if target_normalised not in block_normalised and block_normalised not in target_normalised:
            continue
        
        # Extract Line Items from this block
        line_items_match = re.search(
            r'Line Items: (\[.*?\])',
            block, re.DOTALL
        )
        
        if line_items_match:
            try:
                parsed = json.loads(line_items_match.group(1))
                for item in parsed:
                    code = item.get('code')
                    if code and code not in seen_codes:
                        # Only add items with actual data
                        if item.get('quantity') is not None or item.get('unit_price') is not None:
                            seen_codes.add(code)
                            items.append(item)
            except (json.JSONDecodeError, AttributeError):
                continue
    
    # Fallback: if no typed match found, take first block with items
    if not items:
        for block in result_blocks:
            line_items_match = re.search(r'Line Items: (\[.*?\])', block, re.DOTALL)
            if line_items_match:
                try:
                    parsed = json.loads(line_items_match.group(1))
                    for item in parsed:
                        code = item.get('code')
                        if code and code not in seen_codes:
                            seen_codes.add(code)
                            items.append(item)
                    if items:
                        break
                except (json.JSONDecodeError, AttributeError):
                    continue
    
    return items


def _extract_total_from_search(search_output: str, target_doc_type: str) -> Optional[float]:
    """Extract document total from search results."""
    import re
    
    # Look for total in the target document type section
    pattern = re.search(
        r'Document Type: ' + target_doc_type.replace('_', r'[\s_]') +
        r'.*?Document Total: £([\d.]+)',
        search_output, re.DOTALL | re.IGNORECASE
    )
    
    if pattern:
        try:
            return float(pattern.group(1))
        except ValueError:
            pass
    
    # Fallback
    fallback = re.search(r'Document Total: £([\d.]+)', search_output)
    if fallback:
        try:
            return float(fallback.group(1))
        except ValueError:
            pass
    
    return None


def _extract_ref_from_search(search_output: str, ref_type: str) -> str:
    """Extract a reference number from search results."""
    import re
    
    patterns = {
        'invoice_ref': r'Invoice Ref: (INV-[A-Z]{2,3}-\d{3,4})',
        'dn_ref': r'DN Ref: (DN-\d{3,4})',
        'po_ref': r'PO Ref: (PO-\d{4}-\d{3,4})'
    }
    
    pattern = patterns.get(ref_type, '')
    if pattern:
        match = re.search(pattern, search_output)
        if match:
            return match.group(1)
    
    return 'NOT_FOUND'


def _compare_po_invoice(
    po_ref: str, inv_ref: str,
    po_items: List[Dict], inv_items: List[Dict],
    po_total: Optional[float], inv_total: Optional[float],
    verbose: bool
) -> List[Dict]:
    """Compare PO and Invoice for price and quantity discrepancies."""
    
    anomalies = []
    
    if verbose:
        print(f"\n  Comparing PO vs Invoice...")
    
    # Build lookup dict for invoice items
    inv_lookup = {item['code']: item for item in inv_items}
    
    for po_item in po_items:
        code = po_item.get('code')
        if not code:
            continue
        
        inv_item = inv_lookup.get(code)
        if not inv_item:
            if verbose:
                print(f"    {code}: not found in invoice")
            continue
        
        # Compare unit prices
        po_price = po_item.get('unit_price')
        inv_price = inv_item.get('unit_price')
        
        if po_price is not None and inv_price is not None:
            price_comparison = compare_values.invoke(
                f"unit_price|{po_price}|{inv_price}|{po_ref}|{inv_ref}"
            )
            
            if 'DISCREPANCY' in price_comparison:
                if verbose:
                    print(f"    ✗ PRICE MISMATCH: {code} — PO:£{po_price} vs INV:£{inv_price}")
                
                po_qty = po_item.get('quantity', 0) or 0
                impact_amount = abs(inv_price - po_price) * po_qty
                
                flag_anomaly.invoke(json.dumps({
                    'anomaly_type': 'price_mismatch',
                    'po_ref': po_ref,
                    'document1': f"{po_ref}.pdf",
                    'document2': f"{inv_ref}.pdf",
                    'field': 'unit_price',
                    'item_code': code,
                    'value1': str(po_price),
                    'value2': str(inv_price),
                    'impact': f'£{impact_amount:.2f} overcharge on {po_qty} units of {code}',
                    'severity': 'high' if impact_amount > 100 else 'medium'
                }))
                
                anomalies.append({
                    'type': 'price_mismatch',
                    'item_code': code,
                    'po_value': po_price,
                    'inv_value': inv_price,
                    'details': f'{code}: PO £{po_price} vs Invoice £{inv_price}'
                })
            else:
                if verbose:
                    print(f"    ✓ Price match: {code} — £{po_price}")
        
        # Compare quantities
        po_qty = po_item.get('quantity')
        inv_qty = inv_item.get('quantity')
        
        if po_qty is not None and inv_qty is not None:
            qty_comparison = compare_values.invoke(
                f"quantity|{po_qty}|{inv_qty}|{po_ref}|{inv_ref}"
            )
            
            if 'DISCREPANCY' in qty_comparison:
                if verbose:
                    print(f"    ✗ QUANTITY MISMATCH: {code} — PO:{po_qty} vs INV:{inv_qty}")
                
                flag_anomaly.invoke(json.dumps({
                    'anomaly_type': 'quantity_mismatch',
                    'po_ref': po_ref,
                    'document1': f"{po_ref}.pdf",
                    'document2': f"{inv_ref}.pdf",
                    'field': 'quantity',
                    'item_code': code,
                    'value1': str(po_qty),
                    'value2': str(inv_qty),
                    'impact': f'PO ordered {po_qty} units, Invoice charges for {inv_qty} units of {code}',
                    'severity': 'high'
                }))
                
                anomalies.append({
                    'type': 'quantity_mismatch',
                    'item_code': code,
                    'po_value': po_qty,
                    'inv_value': inv_qty,
                    'details': f'{code}: PO qty {po_qty} vs Invoice qty {inv_qty}'
                })
            else:
                if verbose:
                    print(f"    ✓ Quantity match: {code} — {po_qty} units")
    
    # Compare totals
    if po_total is not None and inv_total is not None:
        total_comparison = compare_values.invoke(
            f"total_amount|{po_total}|{inv_total}|{po_ref}|{inv_ref}"
        )
        
        if 'DISCREPANCY' in total_comparison:
            if verbose:
                print(f"    ✗ TOTAL MISMATCH: PO:£{po_total} vs INV:£{inv_total}")
            
            flag_anomaly.invoke(json.dumps({
                'anomaly_type': 'total_mismatch',
                'po_ref': po_ref,
                'document1': f"{po_ref}.pdf",
                'document2': f"{inv_ref}.pdf",
                'field': 'total_amount',
                'value1': str(po_total),
                'value2': str(inv_total),
                'impact': f'Total discrepancy of £{abs(inv_total - po_total):.2f}',
                'severity': 'high'
            }))
            
            anomalies.append({
                'type': 'total_mismatch',
                'po_value': po_total,
                'inv_value': inv_total,
                'details': f'Total: PO £{po_total} vs Invoice £{inv_total}'
            })
        else:
            if verbose:
                print(f"    ✓ Total match: £{po_total}")
    
    return anomalies


def _compare_po_dn(
    po_ref: str, dn_ref: str,
    po_items: List[Dict], dn_items: List[Dict],
    verbose: bool
) -> List[Dict]:
    """Compare PO and Delivery Note for quantity discrepancies."""
    
    anomalies = []
    
    if verbose:
        print(f"\n  Comparing PO vs Delivery Note...")
    
    # Build lookup dict for DN items
    dn_lookup = {item['code']: item for item in dn_items}
    
    for po_item in po_items:
        code = po_item.get('code')
        if not code:
            continue
        
        dn_item = dn_lookup.get(code)
        if not dn_item:
            if verbose:
                print(f"    {code}: not found in delivery note")
            continue
        
        # Compare quantities
        po_qty = po_item.get('quantity')
        dn_qty = dn_item.get('quantity')
        
        if po_qty is not None and dn_qty is not None:
            qty_comparison = compare_values.invoke(
                f"quantity|{po_qty}|{dn_qty}|{po_ref}|{dn_ref}"
            )
            
            if 'DISCREPANCY' in qty_comparison:
                shortfall = po_qty - dn_qty
                direction = 'short' if shortfall > 0 else 'over-delivery'
                
                if verbose:
                    print(f"    ✗ QUANTITY MISMATCH: {code} — "
                          f"Ordered:{po_qty} vs Delivered:{dn_qty} "
                          f"({abs(shortfall)} units {direction})")
                
                flag_anomaly.invoke(json.dumps({
                    'anomaly_type': 'quantity_mismatch',
                    'po_ref': po_ref,
                    'document1': f"{po_ref}.pdf",
                    'document2': f"{dn_ref}.pdf",
                    'field': 'quantity_delivered',
                    'item_code': code,
                    'value1': str(po_qty),
                    'value2': str(dn_qty),
                    'impact': f'{abs(shortfall)} units {direction} — {code}',
                    'severity': 'medium'
                }))
                
                anomalies.append({
                    'type': 'quantity_mismatch',
                    'item_code': code,
                    'po_value': po_qty,
                    'dn_value': dn_qty,
                    'details': f'{code}: ordered {po_qty} vs delivered {dn_qty}'
                })
            else:
                if verbose:
                    print(f"    ✓ Quantity match: {code} — {po_qty} units delivered")
    
    return anomalies


# ============================================================
# BATCH EVALUATION — Run all document sets
# ============================================================

def run_full_evaluation(verbose: bool = False) -> Dict:
    """
    Run verification on all document sets in the vector store.
    Returns evaluation metrics for dissertation results chapter.
    """
    
    _, metadata, document_results = get_resources()
    
    # Get all unique PO references
    po_refs = sorted(set(
        m['po_ref'] for m in metadata
        if m['po_ref'] != 'NOT_FOUND'
        and not m['po_ref'].startswith('PO-(informal')
    ))
    
    print(f"\nRunning full evaluation on {len(po_refs)} document sets...")
    print("="*60)
    
    evaluation_results = []
    
    for po_ref in po_refs:
        result = verify_document_set(po_ref, verbose=verbose)
        evaluation_results.append(result)
        
        status = "ANOMALY" if not result['consistent'] else "CONSISTENT"
        anomaly_count = len(result['anomalies'])
        print(f"  {po_ref}: {status} ({anomaly_count} anomalies)")
    
    # Save results
    output_path = Path("vector_store/evaluation_results.json")
    with open(output_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    print(f"\nEvaluation complete. Results saved to: {output_path}")
    print(generate_anomaly_report())
    
    return evaluation_results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Load vector store
    get_resources()
    
    print("Supply Chain Document Verification Pipeline")
    print("="*60)
    print("\nOptions:")
    print("1. Verify a single document set")
    print("2. Run full evaluation on all sets")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        po_ref = input("Enter PO reference (e.g. PO-2026-2001): ").strip()
        result = verify_document_set(po_ref, verbose=True)
        print("\n" + generate_anomaly_report())
    
    elif choice == "2":
        results = run_full_evaluation(verbose=False)
    
    else:
        # Default: test on known anomaly sets
        print("\nRunning tests on known anomaly sets...")
        
        test_sets = [
            "PO-2026-2001",  # Set 06: price mismatch
            "PO-2026-2004",  # Set 09: quantity mismatch
        ]
        
        for po_ref in test_sets:
            verify_document_set(po_ref, verbose=True)
        
        print("\n" + generate_anomaly_report())
