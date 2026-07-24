import json

with open('vector_store/metadata.json') as f:
    metadata = json.load(f)

# Check Sets 12, 13, 14 for missing documents
for po_ref in ['PO-2026-2007', 'PO-2026-2008']:
    chunks = [m for m in metadata if m.get('po_ref') == po_ref]
    print(f'\n=== {po_ref} ===')
    print(f'Total chunks: {len(chunks)}')
    doc_types = set(c['document_type'] for c in chunks)
    print(f'Document types found: {doc_types}')
    for c in chunks:
        print(f'  {c["filename"]} | {c["document_type"]} | {c["chunk_type"]}')

# Check if any chunks have po_ref NOT_FOUND that might be Set 14
no_po = [m for m in metadata if m.get('po_ref') == 'NOT_FOUND' or 'NOT_PROVIDED' in str(m.get('po_ref', ''))]
print(f'\n=== Documents with no PO ref: {len(no_po)} ===')
for c in no_po[:5]:
    print(f'  {c["filename"]} | {c["document_type"]} | po_ref: {c["po_ref"]}')
