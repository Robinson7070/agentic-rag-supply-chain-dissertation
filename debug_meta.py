import json

with open('vector_store/metadata.json') as f:
    metadata = json.load(f)

print("=== All chunks for PO-2026-2002 ===")
chunks = [m for m in metadata if m.get('po_ref') == 'PO-2026-2002']
for c in chunks:
    print(f"  {c['filename']} | {c['document_type']} | po_ref:{c['po_ref']} | inv_ref:{c['invoice_ref']}")

print()
print("=== INV-CF-4410 chunks ===")
chunks2 = [m for m in metadata if 'INV-CF-4410' in m.get('filename', '')]
for c in chunks2:
    print(f"  {c['filename']} | {c['document_type']} | po_ref:{c['po_ref']} | inv_ref:{c['invoice_ref']}")

print()
print("=== INV-CF-3389 chunks ===")
chunks3 = [m for m in metadata if 'INV-CF-3389' in m.get('filename', '')]
for c in chunks3:
    print(f"  {c['filename']} | {c['document_type']} | po_ref:{c['po_ref']} | inv_ref:{c['invoice_ref']}")
