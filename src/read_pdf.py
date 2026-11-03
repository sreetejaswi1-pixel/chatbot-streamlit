from pathlib import Path
from pypdf import PdfReader

# Find PDF files in the documents folder
pdf_files = list(Path("documents").glob("*.pdf"))

if not pdf_files:
    print("No PDF found in the documents folder.")
    exit()

# Select the first PDF
pdf_path = pdf_files[0]

print(f"Reading PDF: {pdf_path.name}")

# Read the PDF
reader = PdfReader(pdf_path)

print(f"Number of pages: {len(reader.pages)}")

# Extract text from each page
for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text()

    print(f"\n--- Page {page_number} ---")
    print(text[:1000])