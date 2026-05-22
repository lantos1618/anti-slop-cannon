from decimal import Decimal


def parse_total_rows(rows):
    total = Decimal("0")
    for row in rows:
        if not row:
            continue
        if row.get("voided"):
            continue
        amount = Decimal(str(row.get("amount", "0")))
        tax = Decimal(str(row.get("tax", "0")))
        total += amount + tax
    return total.quantize(Decimal("0.01"))


def render_invoice_summary(rows):
    total = parse_total_rows(rows)
    return {
        "title": "Invoice summary",
        "row_count": len(rows),
        "total": str(total),
    }
