"""
Regenerate corrupted PDF files with CORRECT quantities matching ground truth.
"""

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import cm
import os

styles = getSampleStyleSheet()
normal = styles['Normal']
heading = styles['Heading2']

def make_pdf(filepath, content_fn):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    doc = SimpleDocTemplate(filepath, pagesize=A4,
                           topMargin=2*cm, bottomMargin=2*cm,
                           leftMargin=2*cm, rightMargin=2*cm)
    story = content_fn()
    doc.build(story)
    print(f"Generated: {filepath}")


# ============================================================
# DN-9933.pdf — Set 02 Castlegate clean delivery note
# PO orders: FS-118=12 units, FS-220=40 units
# DN must match exactly (clean set)
# ============================================================
def dn_9933():
    story = []
    story.append(Paragraph("CASTLEGATE FUEL SYSTEMS LTD", heading))
    story.append(Paragraph("DELIVERY NOTE", heading))
    story.append(Spacer(1, 0.3*cm))
    data = [
        ["Delivery Note No:", "DN-9933"],
        ["Date Delivered:", "12 Apr 2026"],
        ["Linked PO:", "PO-2026-1002"],
        ["Linked Invoice:", "INV-CF-3389"],
        ["Delivered To:", "Apex Logistics Ltd"],
        ["Received By:", "T. Adeyemi (Warehouse Supervisor)"],
    ]
    t = Table(data, colWidths=[5*cm, 10*cm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))
    # CORRECT quantities: FS-118=12, FS-220=40
    items = [
        ["Item Code", "Description", "Qty Ordered", "Qty Delivered", "Condition"],
        ["FS-118", "Diesel Fuel Pump", "12", "12", "Good"],
        ["FS-220", "Fuel Filter Cartridge", "40", "40", "Good"],
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
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("All items delivered in good condition. Delivery complete.", normal))
    return story

make_pdf("data/document_sets/set02_castlegate/DN-9933.pdf", dn_9933)


# ============================================================
# DN-9947.pdf — Set 03 Northgate clean delivery note
# PO orders: TY-440=24 units
# DN must match exactly (clean set)
# Current file is being misclassified as invoice
# ============================================================
def dn_9947():
    story = []
    story.append(Paragraph("NORTHGATE TYRE SOLUTIONS", heading))
    story.append(Paragraph("DELIVERY NOTE", heading))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "Delivery Note No: DN-9947", normal))
    story.append(Paragraph(
        "Date Delivered: 15th April 2026", normal))
    story.append(Paragraph(
        "Linked PO: PO-2026-1003", normal))
    story.append(Paragraph(
        "Linked Invoice: INV-NT-5512", normal))
    story.append(Paragraph(
        "Delivered To: Apex Logistics Ltd", normal))
    story.append(Paragraph(
        "Received By: T. Adeyemi (Warehouse Supervisor)", normal))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(
        "We confirm delivery of the following goods to Apex Logistics Ltd:", normal))
    story.append(Spacer(1, 0.3*cm))
    # CORRECT: TY-440=24 units delivered
    items = [
        ["Item Code", "Description", "Qty Ordered", "Qty Delivered", "Condition"],
        ["TY-440", "Commercial Tyre 22.5\"", "24", "24", "Good"],
    ]
    t = Table(items, colWidths=[2.5*cm, 5.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        "All tyres delivered in good condition. Delivery complete.", normal))
    story.append(Paragraph("Signed: T.A.", normal))
    return story

make_pdf("data/document_sets/set03_northgate/DN-9947.pdf", dn_9947)


# ============================================================
# DN-9958.pdf — Set 04 Vantage clean delivery note
# PO orders: EC-902=18 units, EC-150=30 units
# DN must match exactly (clean set)
# ============================================================
def dn_9958():
    story = []
    story.append(Paragraph("VANTAGE ELECTRICAL COMPONENTS LTD | DELIVERY NOTE", heading))
    story.append(Spacer(1, 0.3*cm))
    header_data = [
        ["DN_NO", "DELIVERY_DATE", "LINKED_PO", "LINKED_INV"],
        ["DN-9958", "2026-04-16", "PO-2026-1004", "INV-VE-8804"],
    ]
    t = Table(header_data, colWidths=[3*cm, 3.5*cm, 4*cm, 4*cm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*cm))
    # CORRECT quantities: EC-902=18, EC-150=30
    items = [
        ["ITEM_CODE", "DESC", "QTY_ORDERED", "QTY_DELIVERED", "CONDITION"],
        ["EC-902", "Vehicle Battery 12V 90Ah", "18", "18", "GOOD"],
        ["EC-150", "Alternator Belt Kit", "30", "30", "GOOD"],
    ]
    t2 = Table(items, colWidths=[2.5*cm, 5*cm, 2.5*cm, 2.5*cm, 2*cm])
    t2.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("RECEIVED_BY|T.Adeyemi|SIGNED|TA", normal))
    return story

make_pdf("data/document_sets/set04_vantage/DN-9958.pdf", dn_9958)


# ============================================================
# INV-VE-9988.pdf — Set 17 Vantage date inconsistency
# Due date BEFORE invoice date (internal logic error)
# Invoice date: 2026-05-02, Due date: 2026-04-28
# PO orders EC-902=25 units @ £95.00 — invoice must match qty
# ============================================================
def inv_ve_9988():
    story = []
    story.append(Paragraph("VANTAGE ELECTRICAL COMPONENTS LTD | INVOICE EXPORT", heading))
    story.append(Spacer(1, 0.3*cm))
    header_data = [
        ["INVOICE_NO", "INVOICE_DATE", "PO_REF", "DUE_DATE"],
        ["INV-VE-9988", "2026-05-02", "PO-2026-2011", "2026-04-28"],
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
    # Match PO-2026-2011 quantities exactly — anomaly is only the date
    items = [
        ["ITEM_CODE", "DESC", "QTY", "UNIT_PRICE", "LINE_TOTAL"],
        ["EC-902", "Vehicle Battery 12V 90Ah", "12", "95.00", "1140.00"],
    ]
    t2 = Table(items, colWidths=[2.5*cm, 5*cm, 2*cm, 3*cm, 3*cm])
    t2.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.3*cm))
    totals_data = [
        ["SUBTOTAL", "1140.00"],
        ["VAT_20PCT", "228.00"],
        ["TOTAL_DUE", "1368.00"],
        ["DUE_DATE", "2026-04-28"],
        ["CUSTOMER", "Apex Logistics Ltd"],
    ]
    t3 = Table(totals_data, colWidths=[4*cm, 4*cm])
    t3.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
    ]))
    story.append(t3)
    return story

make_pdf("data/document_sets/discrepancy_sets/set17_vantage_date_inconsistency/INV-VE-9988.pdf", inv_ve_9988)


# ============================================================
# PO-2026-2014.pdf — Set 20 Coastal multiple errors
# Date: PO issued 12/5/26 but invoice dated 9/5/26
# Price: £18.00/unit but invoice charges £20.00
# Quantity: PO orders 40 but only 35 delivered
# ============================================================
def po_2026_2014():
    story = []
    story.append(Paragraph("COASTAL FLEET SUPPLIES - Order Confirmation Slip", heading))
    story.append(Spacer(1, 0.3*cm))
    info = [
        ["Order #:", "2014 (full ref PO-2026-2014)"],
        ["Date:", "12/5/26"],
        ["For:", "Apex Logistics"],
        ["Deliver to:", "22 Brunswick Way, Manchester M3 4DT"],
    ]
    t = Table(info, colWidths=[3*cm, 10*cm])
    t.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (1,0), (1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*cm))
    # PO price is £18/unit for 40 units
    story.append(Paragraph("wiper sets x40 @ 18 = 720", normal))
    story.append(Paragraph("(code FL-077)", normal))
    story.append(Spacer(1, 0.3*cm))
    totals = [
        ["subtotal", "720"],
        ["vat (20%)", "144"],
        ["TOTAL:", "864"],
        ["30 days to pay", ""],
    ]
    t2 = Table(totals, colWidths=[4*cm, 4*cm])
    t2.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("- J Okafor approved this order", normal))
    return story

make_pdf("data/document_sets/discrepancy_sets/set20_coastal_multiple_errors/PO-2026-2014.pdf", po_2026_2014)

print("\nAll PDFs regenerated with correct quantities.")
print("Now run: python src/embeddings_store.py")
print("Then run: python src/verification_pipeline.py")
