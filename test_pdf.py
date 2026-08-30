from fpdf import FPDF
import markdown

md_text = """
# PatientTriage.ai
## Business Proposal

This is a **bold** statement and *italic*.
- Item 1
- Item 2
"""

html = markdown.markdown(md_text)

pdf = FPDF()
pdf.add_page()
pdf.set_font("Times", size=12)
pdf.write_html(html)
pdf.output("test.pdf")
