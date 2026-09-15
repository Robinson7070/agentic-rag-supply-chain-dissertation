"""
Expanded Dataset Generator — 100 Document Sets
Victor Chukwudi Robinson — MSc AI & Data Science, UEL 2026

Generates 100 document sets across 5 supplier formats and 6 anomaly categories.
Previous dataset: 20 sets
Expanded dataset: 100 sets (5x larger)

Distribution:
- Clean/consistent: 25 sets (25%)
- Price mismatch: 15 sets (15%)
- Quantity mismatch: 15 sets (15%)
- Missing document: 15 sets (15%)
- Date inconsistency: 15 sets (15%)
- Multiple errors: 15 sets (15%)
"""

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import cm
import os
import json
import random
from datetime import datetime, timedelta

random.seed(42)
styles = getSampleStyleSheet()
normal = styles['Normal']
heading = styles['Heading2']

OUTPUT_DIR = "data/expanded_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def make_pdf(filepath, content_fn):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    doc = SimpleDocTemplate(filepath, pagesize=A4,
                           topMargin=2*cm, bottomMargin=2*cm,
                           leftMargin=2*cm, rightMargin=2*cm)
    story = content_fn()
    doc.build(story)

def random_date(start_year=2026, base_month=3):
    """Generate a random date in 2026"""
    month = random.randint(base_month, base_month + 2)
    day = random.randint(1, 28)
    return datetime(start_year, min(month, 12), day)

def format_date_uk(d):
    return d.strftime("%d/%m/%Y")

def format_date_iso(d):
    return d.strftime("%Y-%m-%d")

def format_date_written(d):
    months = ['January','February','March','April','May','June',
              'July','August','September','October','November','December']
    return f"{d.day}th {months[d.month-1]} {d.year}"

# Product catalogs per supplier
MERIDIAN_PRODUCTS = [
    ("VP-204", "Brake Pad Set (Front)", 38.00, 58.00),
    ("VP-311", "Oil Filter", 7.50, 12.00),
    ("VP-088", "Air Filter Element", 12.00, 18.00),
    ("VP-445", "Wiper Blade Set", 14.00, 22.00),
    ("VP-201", "Timing Belt Kit", 45.00, 75.00),
]

CASTLEGATE_PRODUCTS = [
    ("FS-118", "Diesel Fuel Pump", 155.00, 210.00),
    ("FS-220", "Fuel Filter Cartridge", 12.00, 18.00),
    ("FS-334", "Fuel Injector", 95.00, 145.00),
    ("FS-089", "Fuel Tank Sensor", 35.00, 55.00),
]

NORTHGATE_PRODUCTS = [
    ("TY-440", 'Commercial Tyre 22.5"', 195.00, 245.00),
    ("TY-380", 'Van Tyre 195/65 R15', 65.00, 95.00),
    ("TY-520", 'HGV Tyre 315/80 R22', 285.00, 345.00),
]

VANTAGE_PRODUCTS = [
    ("EC-902", "Vehicle Battery 12V 90Ah", 85.00, 115.00),
    ("EC-150", "Alternator Belt Kit", 22.00, 38.00),
    ("EC-441", "Starter Motor", 145.00, 195.00),
    ("EC-203", "Alternator 120A", 185.00, 245.00),
]

COASTAL_PRODUCTS = [
    ("FL-077", "Wiper Set", 16.00, 24.00),
    ("FL-112", "Screen Wash 5L", 8.00, 14.00),
    ("FL-203", "Engine Oil 5W-30 5L", 22.00, 32.00),
    ("FL-089", "Antifreeze 5L", 12.00, 18.00),
]

CUSTOMERS = [
    "Apex Logistics Ltd, 22 Brunswick Way, Manchester M3 4DT",
    "Northern Fleet Services, 45 Industrial Park, Leeds LS1 2AB",
    "Midlands Haulage Co, 78 Commerce Street, Birmingham B1 3CD",
    "Eastern Transport Ltd, 12 Dockside Road, Felixstowe IP11 3SX",
]

APPROVERS = ["J. Okafor", "T. Adeyemi", "S. Williams", "K. Thompson", "M. Hassan"]

ground_truth = []
set_counter = 0

# ============================================================
# MERIDIAN FORMAT GENERATORS
# ============================================================

def meridian_po(po_ref, product, qty, unit_price, po_date, customer):
    def content():
        story = []
        story.append(Paragraph("MERIDIAN VEHICLE PARTS LTD", heading))
        story.append(Paragraph("PURCHASE ORDER", heading))
        story.append(Spacer(1, 0.3*cm))
        data = [
            ["PO Number:", po_ref],
            ["Date:", format_date_uk(po_date)],
            ["Supplier:", "Meridian Vehicle Parts Ltd"],
            ["Bill To:", customer],
        ]
        t = Table(data, colWidths=[4*cm, 12*cm])
        t.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.4*cm))
        items = [
            ["No.", "Part Code", "Description", "Qty", "Unit £", "Total £"],
            ["1", product[0], product[1], str(qty), f"{unit_price:.2f}", f"{qty*unit_price:.2f}"],
        ]
        t2 = Table(items, colWidths=[1*cm, 2.5*cm, 6*cm, 1.5*cm, 2*cm, 2*cm])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ]))
        story.append(t2)
        story.append(Spacer(1, 0.3*cm))
        subtotal = qty * unit_price
        vat = subtotal * 0.20
        total = subtotal + vat
        totals = [
            ["Subtotal:", f"£{subtotal:.2f}"],
            ["VAT (20%):", f"£{vat:.2f}"],
            ["ORDER TOTAL:", f"£{total:.2f}"],
        ]
        t3 = Table(totals, colWidths=[4*cm, 4*cm])
        t3.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
        ]))
        story.append(t3)
        story.append(Paragraph(f"Approved by: {random.choice(APPROVERS)}", normal))
        return story
    return content

def meridian_invoice(inv_ref, po_ref, product, qty, unit_price, inv_date, customer):
    def content():
        story = []
        story.append(Paragraph("MERIDIAN VEHICLE PARTS LTD", heading))
        story.append(Paragraph(f"INVOICE {inv_ref}", heading))
        story.append(Spacer(1, 0.3*cm))
        data = [
            ["Invoice No:", inv_ref],
            ["Date:", format_date_uk(inv_date)],
            ["Your ref (PO):", po_ref],
            ["Billed to:", customer],
        ]
        t = Table(data, colWidths=[4*cm, 12*cm])
        t.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.4*cm))
        items = [
            ["No.", "Part Code", "Description", "Qty", "Unit £", "Total £"],
            ["1", product[0], product[1], str(qty), f"{unit_price:.2f}", f"{qty*unit_price:.2f}"],
        ]
        t2 = Table(items, colWidths=[1*cm, 2.5*cm, 6*cm, 1.5*cm, 2*cm, 2*cm])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ]))
        story.append(t2)
        story.append(Spacer(1, 0.3*cm))
        subtotal = qty * unit_price
        vat = subtotal * 0.20
        total = subtotal + vat
        totals = [
            ["Subtotal:", f"£{subtotal:.2f}"],
            ["VAT (20%):", f"£{vat:.2f}"],
            ["TOTAL PAYABLE:", f"£{total:.2f}"],
        ]
        t3 = Table(totals, colWidths=[4*cm, 4*cm])
        t3.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
        ]))
        story.append(t3)
        due = inv_date + timedelta(days=30)
        story.append(Paragraph(f"Payment due by: {format_date_uk(due)}", normal))
        return story
    return content

def meridian_dn(dn_ref, po_ref, inv_ref, product, qty, del_date, customer):
    def content():
        story = []
        story.append(Paragraph("MERIDIAN VEHICLE PARTS LTD", heading))
        story.append(Paragraph("DELIVERY NOTE", heading))
        story.append(Spacer(1, 0.3*cm))
        data = [
            ["Delivery Note:", dn_ref],
            ["Date Delivered:", format_date_uk(del_date)],
            ["Linked PO:", po_ref],
            ["Linked Invoice:", inv_ref],
            ["Delivered To:", customer],
            ["Received By:", f"{random.choice(APPROVERS)} (Warehouse Supervisor)"],
        ]
        t = Table(data, colWidths=[4*cm, 12*cm])
        t.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.4*cm))
        items = [
            ["Part Code", "Description", "Qty Ordered", "Qty Delivered", "Condition"],
            [product[0], product[1], str(qty), str(qty), "Good"],
        ]
        t2 = Table(items, colWidths=[2.5*cm, 5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ]))
        story.append(t2)
        story.append(Paragraph("All items delivered in good condition.", normal))
        return story
    return content

# ============================================================
# VANTAGE FORMAT GENERATORS
# ============================================================

def vantage_po(po_ref, product, qty, unit_price, po_date, customer):
    def content():
        story = []
        story.append(Paragraph("VANTAGE ELECTRICAL COMPONENTS LTD | PURCHASE ORDER", heading))
        story.append(Spacer(1, 0.3*cm))
        header_data = [
            ["PO_NO", "PO_DATE", "CUSTOMER", "PAYMENT_TERMS"],
            [po_ref, format_date_iso(po_date), customer.split(',')[0], "NET30"],
        ]
        t = Table(header_data, colWidths=[3.5*cm, 3*cm, 6*cm, 3*cm])
        t.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3*cm))
        items = [
            ["ITEM_CODE", "DESC", "QTY", "UNIT_PRICE", "LINE_TOTAL"],
            [product[0], product[1], str(qty), f"{unit_price:.2f}", f"{qty*unit_price:.2f}"],
        ]
        t2 = Table(items, colWidths=[2.5*cm, 5*cm, 1.5*cm, 3*cm, 3*cm])
        t2.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ]))
        story.append(t2)
        story.append(Spacer(1, 0.3*cm))
        subtotal = qty * unit_price
        vat = subtotal * 0.20
        total = subtotal + vat
        totals = [
            ["SUBTOTAL", f"{subtotal:.2f}"],
            ["VAT_20PCT", f"{vat:.2f}"],
            ["ORDER_TOTAL", f"{total:.2f}"],
        ]
        t3 = Table(totals, colWidths=[4*cm, 4*cm])
        t3.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
        ]))
        story.append(t3)
        return story
    return content

def vantage_invoice(inv_ref, po_ref, product, qty, unit_price, inv_date, due_date, customer):
    def content():
        story = []
        story.append(Paragraph("VANTAGE ELECTRICAL COMPONENTS LTD | INVOICE EXPORT", heading))
        story.append(Spacer(1, 0.3*cm))
        header_data = [
            ["INVOICE_NO", "INVOICE_DATE", "PO_REF", "DUE_DATE"],
            [inv_ref, format_date_iso(inv_date), po_ref, format_date_iso(due_date)],
        ]
        t = Table(header_data, colWidths=[3.5*cm, 3.5*cm, 4*cm, 3.5*cm])
        t.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3*cm))
        items = [
            ["ITEM_CODE", "DESC", "QTY", "UNIT_PRICE", "LINE_TOTAL"],
            [product[0], product[1], str(qty), f"{unit_price:.2f}", f"{qty*unit_price:.2f}"],
        ]
        t2 = Table(items, colWidths=[2.5*cm, 5*cm, 1.5*cm, 3*cm, 3*cm])
        t2.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ]))
        story.append(t2)
        story.append(Spacer(1, 0.3*cm))
        subtotal = qty * unit_price
        vat = subtotal * 0.20
        total = subtotal + vat
        totals_data = [
            ["SUBTOTAL", f"{subtotal:.2f}"],
            ["VAT_20PCT", f"{vat:.2f}"],
            ["TOTAL_DUE", f"{total:.2f}"],
            ["DUE_DATE", format_date_iso(due_date)],
            ["CUSTOMER", customer.split(',')[0]],
        ]
        t3 = Table(totals_data, colWidths=[4*cm, 4*cm])
        t3.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
        ]))
        story.append(t3)
        return story
    return content

def vantage_dn(dn_ref, po_ref, product, qty, del_date):
    def content():
        story = []
        story.append(Paragraph("VANTAGE ELECTRICAL COMPONENTS LTD | DELIVERY NOTE", heading))
        story.append(Spacer(1, 0.3*cm))
        header_data = [
            ["DN_NO", "DELIVERY_DATE", "LINKED_PO"],
            [dn_ref, format_date_iso(del_date), po_ref],
        ]
        t = Table(header_data, colWidths=[3*cm, 4*cm, 5*cm])
        t.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3*cm))
        items = [
            ["ITEM_CODE", "DESC", "QTY_ORDERED", "QTY_DELIVERED", "CONDITION"],
            [product[0], product[1], str(qty), str(qty), "GOOD"],
        ]
        t2 = Table(items, colWidths=[2.5*cm, 5*cm, 2.5*cm, 2.5*cm, 2*cm])
        t2.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ]))
        story.append(t2)
        story.append(Paragraph(f"RECEIVED_BY|{random.choice(APPROVERS)}|SIGNED|OK", normal))
        return story
    return content

# ============================================================
# DOCUMENT SET GENERATOR
# ============================================================

def generate_set(set_num, anomaly_type, supplier_format):
    global set_counter
    set_counter += 1
    
    po_num = 3000 + set_num
    po_ref = f"PO-2026-{po_num}"
    
    # Select product based on format
    if supplier_format == "meridian":
        product = random.choice(MERIDIAN_PRODUCTS)
        inv_prefix = "INV-MV"
        dn_prefix = "DN"
        folder = f"set{set_num:03d}_meridian_{anomaly_type}"
    elif supplier_format == "vantage":
        product = random.choice(VANTAGE_PRODUCTS)
        inv_prefix = "INV-VE"
        dn_prefix = "DN"
        folder = f"set{set_num:03d}_vantage_{anomaly_type}"
    elif supplier_format == "castlegate":
        product = random.choice(CASTLEGATE_PRODUCTS)
        inv_prefix = "INV-CF"
        dn_prefix = "DN"
        folder = f"set{set_num:03d}_castlegate_{anomaly_type}"
    elif supplier_format == "northgate":
        product = random.choice(NORTHGATE_PRODUCTS)
        inv_prefix = "INV-NT"
        dn_prefix = "DN"
        folder = f"set{set_num:03d}_northgate_{anomaly_type}"
    else:  # coastal
        product = random.choice(COASTAL_PRODUCTS)
        inv_prefix = "INV-CFS"
        dn_prefix = "DN"
        folder = f"set{set_num:03d}_coastal_{anomaly_type}"
    
    inv_num = 5000 + set_num
    dn_num = 7000 + set_num
    inv_ref = f"{inv_prefix}-{inv_num}"
    dn_ref = f"{dn_prefix}-{dn_num}"
    
    customer = random.choice(CUSTOMERS)
    qty = random.randint(10, 50)
    base_price = round(random.uniform(product[2], product[3]), 2)
    
    # Generate dates
    po_date = random_date()
    inv_date = po_date + timedelta(days=random.randint(3, 10))
    del_date = inv_date + timedelta(days=random.randint(2, 7))
    due_date = inv_date + timedelta(days=30)
    
    set_path = os.path.join(OUTPUT_DIR, folder)
    os.makedirs(set_path, exist_ok=True)
    
    gt_entry = {
        "set_num": set_num,
        "po_ref": po_ref,
        "supplier_format": supplier_format,
        "anomaly_type": anomaly_type,
        "product_code": product[0],
        "base_qty": qty,
        "base_price": base_price,
        "has_anomaly": anomaly_type != "consistent",
        "anomaly_description": None,
        "financial_impact": 0.0
    }
    
    # Apply anomaly
    inv_qty = qty
    inv_price = base_price
    dn_qty = qty
    missing_doc = None
    
    if anomaly_type == "price_mismatch":
        price_delta = round(random.uniform(2.0, 15.0), 2)
        inv_price = base_price + price_delta
        impact = price_delta * qty
        gt_entry["anomaly_description"] = f"{product[0]}: PO £{base_price} vs Invoice £{inv_price}"
        gt_entry["financial_impact"] = round(impact, 2)
    
    elif anomaly_type == "quantity_mismatch":
        shortfall = random.randint(2, min(8, qty-1))
        dn_qty = qty - shortfall
        impact = shortfall * base_price
        gt_entry["anomaly_description"] = f"{product[0]}: ordered {qty}, delivered {dn_qty}"
        gt_entry["financial_impact"] = round(impact, 2)
    
    elif anomaly_type == "missing_document":
        missing_doc = random.choice(["invoice", "delivery_note"])
        gt_entry["anomaly_description"] = f"Missing {missing_doc}"
    
    elif anomaly_type == "date_inconsistency":
        date_type = random.choice(["invoice_before_po", "delivery_before_po", "due_before_invoice"])
        if date_type == "invoice_before_po":
            inv_date = po_date - timedelta(days=random.randint(2, 7))
            gt_entry["anomaly_description"] = f"Invoice dated {format_date_iso(inv_date)} before PO {format_date_iso(po_date)}"
        elif date_type == "delivery_before_po":
            del_date = po_date - timedelta(days=random.randint(2, 5))
            gt_entry["anomaly_description"] = f"Delivery dated {format_date_iso(del_date)} before PO {format_date_iso(po_date)}"
        else:
            due_date = inv_date - timedelta(days=random.randint(2, 5))
            gt_entry["anomaly_description"] = f"Due date {format_date_iso(due_date)} before invoice {format_date_iso(inv_date)}"
    
    elif anomaly_type == "multiple_errors":
        price_delta = round(random.uniform(3.0, 12.0), 2)
        inv_price = base_price + price_delta
        shortfall = random.randint(2, min(5, qty-1))
        dn_qty = qty - shortfall
        impact = (price_delta * qty) + (shortfall * base_price)
        gt_entry["anomaly_description"] = f"Price mismatch AND quantity mismatch"
        gt_entry["financial_impact"] = round(impact, 2)
    
    # Generate documents
    if supplier_format == "meridian":
        if missing_doc != "purchase_order":
            make_pdf(
                os.path.join(set_path, f"{po_ref}.pdf"),
                meridian_po(po_ref, product, qty, base_price, po_date, customer)
            )
        if missing_doc != "invoice":
            make_pdf(
                os.path.join(set_path, f"{inv_ref}.pdf"),
                meridian_invoice(inv_ref, po_ref, product, inv_qty, inv_price, inv_date, customer)
            )
        if missing_doc != "delivery_note":
            make_pdf(
                os.path.join(set_path, f"{dn_ref}.pdf"),
                meridian_dn(dn_ref, po_ref, inv_ref, product, dn_qty, del_date, customer)
            )
    
    elif supplier_format == "vantage":
        if missing_doc != "purchase_order":
            make_pdf(
                os.path.join(set_path, f"{po_ref}.pdf"),
                vantage_po(po_ref, product, qty, base_price, po_date, customer)
            )
        if missing_doc != "invoice":
            make_pdf(
                os.path.join(set_path, f"{inv_ref}.pdf"),
                vantage_invoice(inv_ref, po_ref, product, inv_qty, inv_price, inv_date, due_date, customer)
            )
        if missing_doc != "delivery_note":
            make_pdf(
                os.path.join(set_path, f"{dn_ref}.pdf"),
                vantage_dn(dn_ref, po_ref, product, dn_qty, del_date)
            )
    
    else:
        # Use meridian format as fallback for castlegate/northgate/coastal
        # (in a full implementation each would have their own format)
        if missing_doc != "purchase_order":
            make_pdf(
                os.path.join(set_path, f"{po_ref}.pdf"),
                meridian_po(po_ref, product, qty, base_price, po_date, customer)
            )
        if missing_doc != "invoice":
            make_pdf(
                os.path.join(set_path, f"{inv_ref}.pdf"),
                meridian_invoice(inv_ref, po_ref, product, inv_qty, inv_price, inv_date, customer)
            )
        if missing_doc != "delivery_note":
            make_pdf(
                os.path.join(set_path, f"{dn_ref}.pdf"),
                meridian_dn(dn_ref, po_ref, inv_ref, product, dn_qty, del_date, customer)
            )
    
    print(f"  Set {set_num:03d}: {po_ref} | {supplier_format} | {anomaly_type}")
    return gt_entry

# ============================================================
# GENERATE ALL 100 SETS
# ============================================================

print("Generating expanded dataset: 100 document sets")
print("="*60)

formats = ["meridian", "vantage", "castlegate", "northgate", "coastal"]
anomaly_types = {
    "consistent": 25,
    "price_mismatch": 15,
    "quantity_mismatch": 15,
    "missing_document": 15,
    "date_inconsistency": 15,
    "multiple_errors": 15,
}

set_num = 1
for anomaly_type, count in anomaly_types.items():
    print(f"\nGenerating {count} {anomaly_type} sets...")
    for i in range(count):
        supplier = formats[i % len(formats)]
        gt_entry = generate_set(set_num, anomaly_type, supplier)
        ground_truth.append(gt_entry)
        set_num += 1

# Save ground truth
with open(os.path.join(OUTPUT_DIR, "ground_truth_expanded.json"), 'w') as f:
    json.dump(ground_truth, f, indent=2)

print("\n" + "="*60)
print(f"COMPLETE: Generated {len(ground_truth)} document sets")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Ground truth: {OUTPUT_DIR}/ground_truth_expanded.json")

# Summary statistics
anomaly_counts = {}
for entry in ground_truth:
    t = entry['anomaly_type']
    anomaly_counts[t] = anomaly_counts.get(t, 0) + 1

print("\nDistribution:")
for t, c in anomaly_counts.items():
    print(f"  {t}: {c} sets")

total_pdfs = sum(
    2 if entry['anomaly_type'] == 'missing_document' else 3
    for entry in ground_truth
)
print(f"\nTotal PDF documents generated: ~{total_pdfs}")
print("\nNext step: Run python src/embeddings_store.py --data-dir data/expanded_dataset")
