import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table
from reportlab.lib.styles import getSampleStyleSheet


def generate_dar(scores_csv: str, output_pdf: str):
    df = pd.read_csv(scores_csv)
    doc = SimpleDocTemplate(output_pdf)
    styles = getSampleStyleSheet()
    elems = []
    elems.append(Paragraph('DAR Report', styles['Title']))
    data = [df.columns.tolist()] + df.values.tolist()
    elems.append(Table(data))
    doc.build(elems)
    print(f"DAR PDF: {output_pdf}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('scores_csv')
    parser.add_argument('pdf_out')
    args = parser.parse_args()
    generate_dar(args.scores_csv, args.pdf_out)