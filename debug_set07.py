import sys, json
sys.path.insert(0, 'src')
from agent import get_resources, search_documents
from verification_pipeline import _extract_items_from_search

get_resources()

print("=== SET 07 DEBUG ===")
print()

# What does searching for PO-2026-2002 invoice return?
inv_search = search_documents.invoke("PO-2026-2002 invoice unit price line items amount due")
print("INVOICE SEARCH RESULT (first 1000 chars):")
print(inv_search[:1000])
print()

# What items does extract find?
inv_items = _extract_items_from_search(inv_search, 'invoice')
print(f"Extracted invoice items: {inv_items}")
print()

# What does PO search return?
po_search = search_documents.invoke("PO-2026-2002 purchase order line items quantities prices")
print("PO SEARCH RESULT (first 500 chars):")
print(po_search[:500])
print()

po_items = _extract_items_from_search(po_search, 'purchase_order')
print(f"Extracted PO items: {po_items}")
