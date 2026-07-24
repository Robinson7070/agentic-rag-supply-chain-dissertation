import json

with open('vector_store/metadata.json') as f:
    metadata = json.load(f)

# Check all clean sets
for po_ref in ['PO-2026-1001', 'PO-2026-1002', 'PO-2026-1003', 'PO-2026-1004', 'PO-2026-1005']:
    chunks = [m for m in metadata if m.get('po_ref') == po_ref]
    doc_types = set(c['document_type'] for c in chunks)
    filenames = set(c['filename'] for c in chunks)
    print(f"{po_ref}: {doc_types} | {filenames}")
