"""
Regenerate corrupted PDF files:
- DN-9933.pdf (Set 02 Castlegate clean)
- DN-9958.pdf (Set 04 Vantage clean)
- INV-VE-9988.pdf (Set 17 Vantage date inconsistency)
- PO-2026-2014.pdf (Set 20 Coastal multiple errors)
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
    items = [
        ["Item Code", "Description", "Qty Ordered", "Qty Delivered", "Condition"],
        ["FS-118", "Diesel Fuel Pump", "10", "10", "Good"],
        ["FS-220", "Fuel Filter Housing", "15", "15", "Good"],
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
    story.append(Paragraph("All items delivered in good condition.", normal))
    return story

make_pdf("data/document_sets/set02_castlegate/DN-9933.pdf", dn_9933)


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
    items = [
        ["ITEM_CODE", "DESC", "QTY_ORDERED", "QTY_DELIVERED", "CONDITION"],
        ["EC-902", "Vehicle Battery 12V 90Ah", "20", "20", "GOOD"],
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
    items = [
        ["ITEM_CODE", "DESC", "QTY", "UNIT_PRICE", "LINE_TOTAL"],
        ["EC-902", "Vehicle Battery 12V 90Ah", "25", "95.00", "2375.00"],
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
        ["SUBTOTAL", "2375.00"],
        ["VAT_20PCT", "475.00"],
        ["TOTAL_DUE", "2850.00"],
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

print("\nAll 4 PDFs regenerated. Now run: python src/embeddings_store.py")
