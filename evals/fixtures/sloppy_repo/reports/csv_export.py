import csv
from io import StringIO


def export_rows_to_csv(rows):
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=["id", "name", "amount"])
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "id": row.get("id", ""),
                "name": row.get("name", ""),
                "amount": row.get("amount", ""),
            }
        )
    return output.getvalue()
