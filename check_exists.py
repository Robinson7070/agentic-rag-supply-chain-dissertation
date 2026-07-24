import json
import sys
sys.path.insert(0, 'src')
from agent import get_resources, check_document_exists

get_resources()

# Test missing document detection
tests = [
    ('PO-2026-2007', 'delivery_note'),   # Set 12 - DN missing
    ('PO-2026-2007', 'invoice'),          # Set 12 - Invoice present
    ('PO-2026-2008', 'invoice'),          # Set 13 - Invoice missing
    ('PO-2026-2008', 'delivery_note'),    # Set 13 - DN present
]

for po_ref, doc_type in tests:
    result = check_document_exists.invoke(f"{po_ref}|{doc_type}")
    print(f"{po_ref} | {doc_type}:")
    print(f"  {result}")
    print()
