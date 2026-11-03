from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# File containing our chunks
chunks_file = Path("results/chunks.txt")

# Read chunks
text = chunks_file.read_text(encoding="utf-8")

# Split into individual chunks
chunks = text.split("--- CHUNK ")[1:]

clean_chunks = []

for chunk in chunks:
    parts = chunk.split("---", 1)

    if len(parts) == 2:
        chunk_text = parts[1].strip()

        if chunk_text:
            clean_chunks.append(chunk_text)

print(f"Loaded {len(clean_chunks)} chunks")

# Load embedding model
print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
print("Creating embeddings...")

embeddings = model.encode(
    clean_chunks,
    show_progress_bar=True
)

# Convert to NumPy
embeddings = np.array(embeddings).astype("float32")

# Create FAISS vector database
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

# Save vector database
Path("results/vector_store").mkdir(exist_ok=True)

faiss.write_index(
    index,
    "results/vector_store/index.faiss"
)

# Save chunks separately
with open(
    "results/vector_store/chunks.txt",
    "w",
    encoding="utf-8"
) as f:

    for i, chunk in enumerate(clean_chunks):
        f.write(f"--- CHUNK {i} ---\n")
        f.write(chunk)
        f.write("\n")

print()
print("SUCCESS!")
print(f"Vectors created: {index.ntotal}")
print(f"Vector dimension: {dimension}")
print("Vector database saved in results/vector_store/")