from pathlib import Path
from pypdf import PdfReader

# PDF location
pdf_path = Path("documents/Generative_AI.pdf")

# Read PDF
reader = PdfReader(pdf_path)

# Extract all text
text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"

# Chunk settings
chunk_size = 1000
overlap = 200

chunks = []

start = 0

while start < len(text):
    end = start + chunk_size
    chunk = text[start:end]

    if chunk.strip():
        chunks.append(chunk.strip())

    start += chunk_size - overlap

# Save chunks
output_path = Path("results/chunks.txt")

with open(output_path, "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks, 1):
        f.write(f"\n--- CHUNK {i} ---\n")
        f.write(chunk)
        f.write("\n")

print("PDF processed successfully!")
print(f"Total characters: {len(text)}")
print(f"Total chunks: {len(chunks)}")
print(f"Chunks saved to: {output_path}")