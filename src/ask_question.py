from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer

# Load the vector database
index = faiss.read_index(
    "results/vector_store/index.faiss"
)

# Load the chunks
chunks_text = Path(
    "results/vector_store/chunks.txt"
).read_text(encoding="utf-8")

# Split saved chunks
chunks = chunks_text.split("--- CHUNK ")

clean_chunks = []

for chunk in chunks:
    if "---" in chunk:
        chunk_text = chunk.split("---", 1)[1].strip()

        if chunk_text:
            clean_chunks.append(chunk_text)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("RAG Question Answering System")
print("--------------------------------")
print("Type 'exit' to quit.")

while True:

    question = input("\nAsk a question: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    # Convert question into embedding
    question_embedding = model.encode(
        [question]
    ).astype("float32")

    # Search vector database
    distances, indices = index.search(
        question_embedding,
        k=3
    )

    print("\nRelevant information from the PDF:")
    print("--------------------------------")

    for i, index_number in enumerate(indices[0]):

        if index_number < len(clean_chunks):

            print(f"\nResult {i + 1}:")
            print(clean_chunks[index_number])

            print(
                f"\nSimilarity distance: "
                f"{distances[0][i]:.4f}"
            )