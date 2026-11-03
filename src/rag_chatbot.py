import os
import sys

import numpy as np
import faiss
import ollama

from sentence_transformers import SentenceTransformer


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

VECTOR_STORE_DIR = os.path.join(
    BASE_DIR,
    "results",
    "vector_store"
)

INDEX_PATH = os.path.join(
    VECTOR_STORE_DIR,
    "index.faiss"
)

CHUNKS_PATH = os.path.join(
    VECTOR_STORE_DIR,
    "chunks.txt"
)


# ============================================================
# 2. CHECK FILES
# ============================================================

print("\nChecking RAG files...")

if not os.path.exists(INDEX_PATH):
    print("ERROR: index.faiss not found.")
    print("Expected location:")
    print(INDEX_PATH)
    sys.exit(1)

if not os.path.exists(CHUNKS_PATH):
    print("ERROR: chunks.txt not found.")
    print("Expected location:")
    print(CHUNKS_PATH)
    sys.exit(1)

print("index.faiss found")
print("chunks.txt found")


# ============================================================
# 3. LOAD FAISS INDEX
# ============================================================

print("\nLoading FAISS vector database...")

try:
    index = faiss.read_index(INDEX_PATH)
except Exception as error:
    print("ERROR loading FAISS index:")
    print(error)
    sys.exit(1)

print("FAISS index loaded.")
print("Number of vectors:", index.ntotal)


# ============================================================
# 4. LOAD TEXT CHUNKS
# ============================================================

print("\nLoading PDF chunks...")

try:
    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        chunks_text = file.read()

except Exception as error:
    print("ERROR reading chunks.txt:")
    print(error)
    sys.exit(1)


# ============================================================
# 5. SPLIT CHUNKS
# ============================================================

chunks = []

# The embedding script created 113 chunks.
# We first try to identify the chunk separators correctly.

if "--- CHUNK " in chunks_text:

    raw_chunks = chunks_text.split("--- CHUNK ")

    for chunk in raw_chunks:

        chunk = chunk.strip()

        if not chunk:
            continue

        lines = chunk.split("\n")

        # Remove the chunk number/header
        if len(lines) > 1:
            chunk_text = "\n".join(lines[1:]).strip()
        else:
            chunk_text = chunk

        if chunk_text:
            chunks.append(chunk_text)

elif "\n---\n" in chunks_text:

    chunks = [
        chunk.strip()
        for chunk in chunks_text.split("\n---\n")
        if chunk.strip()
    ]

else:

    chunks = [
        chunk.strip()
        for chunk in chunks_text.split("\n\n")
        if chunk.strip()
    ]


print("Chunks loaded:", len(chunks))


# ============================================================
# 6. MATCH VECTOR AND CHUNK COUNTS
# ============================================================

if index.ntotal != len(chunks):

    print("\nWARNING:")
    print(
        "FAISS vectors and text chunks have different counts."
    )

    print("FAISS vectors:", index.ntotal)
    print("Text chunks:", len(chunks))

    usable_count = min(
        index.ntotal,
        len(chunks)
    )

    print(
        "Using the first",
        usable_count,
        "matching chunks."
    )

    chunks = chunks[:usable_count]

else:

    print("\nFAISS vectors and text chunks match.")
    print("Total matching chunks:", len(chunks))


# ============================================================
# 7. LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

try:

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

except Exception as error:

    print("ERROR loading embedding model:")
    print(error)
    sys.exit(1)

print("Embedding model loaded.")


# ============================================================
# 8. CHECK OLLAMA
# ============================================================

print("\nChecking Ollama...")

try:

    models = ollama.list()

    print("Ollama is running.")

except Exception as error:

    print("\nERROR: Ollama is not running.")

    print(
        "Please start Ollama first."
    )

    print(
        "You can run:"
    )

    print(
        "ollama serve"
    )

    print("\nDetails:")
    print(error)

    sys.exit(1)


# ============================================================
# 9. CHECK LLAMA MODEL
# ============================================================

MODEL_NAME = "llama3.2:3b"

installed_models = []

for model_info in models["models"]:

    model_name = model_info["model"]

    installed_models.append(
        model_name
    )


if MODEL_NAME not in installed_models:

    print(
        f"\nERROR: {MODEL_NAME} is not installed."
    )

    print("\nRun this command:")

    print(
        f"ollama pull {MODEL_NAME}"
    )

    sys.exit(1)


print(
    f"Ollama model found: {MODEL_NAME}"
)


# ============================================================
# 10. SEARCH DOCUMENTS
# ============================================================

def search_documents(
    question,
    top_k=5
):

    # Create embedding for the question

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )

    # Convert to float32 for FAISS

    question_embedding = np.asarray(
        question_embedding,
        dtype=np.float32
    )

    # Make sure top_k does not exceed
    # the number of vectors

    actual_top_k = min(
        top_k,
        index.ntotal
    )

    # Search FAISS

    distances, indices = index.search(
        question_embedding,
        actual_top_k
    )

    relevant_chunks = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx < 0:
            continue

        if idx >= len(chunks):
            continue

        relevant_chunks.append(
            {
                "text": chunks[idx],
                "distance": float(distance),
                "index": int(idx)
            }
        )

    return relevant_chunks


# ============================================================
# 11. ASK LOCAL LLM
# ============================================================

def ask_llm(
    question,
    relevant_chunks
):

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []

    for i, result in enumerate(
        relevant_chunks,
        start=1
    ):

        context_parts.append(
            f"""
--- Retrieved PDF Context {i} ---

{result["text"]}
"""
        )

    context = "\n".join(
        context_parts
    )


    # --------------------------------------------------------
    # Create prompt
    # --------------------------------------------------------

    prompt = f"""
You are a question-answering assistant for a
course technical document.

Answer the user's question using ONLY the PDF
context provided below.

Do not use outside knowledge.

If the answer cannot be found in the PDF context,
say exactly:

"I could not find that information in the PDF."

Do not invent information.

Give a clear and concise answer.

PDF CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""


    # --------------------------------------------------------
    # Call Llama through Ollama
    # --------------------------------------------------------

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# ============================================================
# 12. START CHATBOT
# ============================================================

print("\n")
print("=" * 60)
print("RAG PDF QUESTION ANSWERING SYSTEM")
print("=" * 60)
print("PDF: Generative_AI.pdf")
print("LLM: Llama 3.2 3B")
print("Vector Database: FAISS")
print("Embedding Model: all-MiniLM-L6-v2")
print("Type 'exit' to quit.")
print("=" * 60)


# ============================================================
# 13. MAIN CHAT LOOP
# ============================================================

while True:

    try:

        question = input(
            "\nAsk a question: "
        ).strip()

    except KeyboardInterrupt:

        print("\n\nExiting...")
        break

    except EOFError:

        print("\n\nExiting...")
        break


    # --------------------------------------------------------
    # Empty question
    # --------------------------------------------------------

    if not question:

        print(
            "Please enter a question."
        )

        continue


    # --------------------------------------------------------
    # Exit
    # --------------------------------------------------------

    if question.lower() in [
        "exit",
        "quit",
        "q"
    ]:

        print("\nGoodbye!")
        break


    # --------------------------------------------------------
    # Search PDF
    # --------------------------------------------------------

    print(
        "\nSearching the PDF..."
    )

    try:

        relevant_chunks = search_documents(
            question,
            top_k=5
        )

    except Exception as error:

        print(
            "\nERROR during document search:"
        )

        print(error)

        continue


    # --------------------------------------------------------
    # Check results
    # --------------------------------------------------------

    if not relevant_chunks:

        print(
            "\nNo relevant information found."
        )

        continue


    print(
        "Retrieved",
        len(relevant_chunks),
        "relevant chunks."
    )


    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    print(
        "Generating answer using Llama 3.2..."
    )

    try:

        answer = ask_llm(
            question,
            relevant_chunks
        )

    except Exception as error:

        print(
            "\nERROR from Ollama:"
        )

        print(error)

        continue


    # --------------------------------------------------------
    # Display answer
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("AI ANSWER")
    print("=" * 60)

    print(answer)

    print("=" * 60)