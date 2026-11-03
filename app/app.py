import os

import faiss
import numpy as np
import ollama
import streamlit as st

from sentence_transformers import SentenceTransformer


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

# app.py is inside:
# rag-qa-project/app/
#
# We need to go up one level to:
# rag-qa-project/

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
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

MODEL_NAME = "llama3.2:3b"


# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Course Document RAG Q&A",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# 3. TITLE
# ============================================================

st.title("📚 Course Document RAG Q&A System")

st.write(
    "Ask questions about the course or technical document."
)

st.info(
    "This system uses FAISS for document retrieval "
    "and Llama 3.2 running locally through Ollama."
)


# ============================================================
# 4. CHECK REQUIRED FILES
# ============================================================

if not os.path.exists(INDEX_PATH):

    st.error(
        "FAISS index not found."
    )

    st.write(
        "Expected file:"
    )

    st.code(
        INDEX_PATH
    )

    st.stop()


if not os.path.exists(CHUNKS_PATH):

    st.error(
        "chunks.txt not found."
    )

    st.write(
        "Expected file:"
    )

    st.code(
        CHUNKS_PATH
    )

    st.stop()


# ============================================================
# 5. LOAD FAISS INDEX
# ============================================================

@st.cache_resource
def load_faiss():

    return faiss.read_index(
        INDEX_PATH
    )


try:

    index = load_faiss()

except Exception as error:

    st.error(
        "Error loading FAISS index."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# 6. LOAD PDF CHUNKS
# ============================================================

@st.cache_data
def load_chunks():

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        chunks_text = file.read()


    chunks = []


    # --------------------------------------------------------
    # Format 1: --- CHUNK 1 ---
    # --------------------------------------------------------

    if "--- CHUNK " in chunks_text:

        raw_chunks = chunks_text.split(
            "--- CHUNK "
        )

        for chunk in raw_chunks:

            chunk = chunk.strip()

            if not chunk:
                continue

            lines = chunk.split("\n")

            if len(lines) > 1:

                chunk_text = "\n".join(
                    lines[1:]
                ).strip()

            else:

                chunk_text = chunk


            if chunk_text:

                chunks.append(
                    chunk_text
                )


    # --------------------------------------------------------
    # Format 2: --- separator ---
    # --------------------------------------------------------

    elif "\n---\n" in chunks_text:

        chunks = [
            chunk.strip()
            for chunk in chunks_text.split(
                "\n---\n"
            )
            if chunk.strip()
        ]


    # --------------------------------------------------------
    # Format 3: blank-line separation
    # --------------------------------------------------------

    else:

        chunks = [
            chunk.strip()
            for chunk in chunks_text.split(
                "\n\n"
            )
            if chunk.strip()
        ]


    return chunks


try:

    chunks = load_chunks()

except Exception as error:

    st.error(
        "Error loading PDF chunks."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# 7. MATCH FAISS AND CHUNK COUNTS
# ============================================================

usable_count = min(
    index.ntotal,
    len(chunks)
)

chunks = chunks[:usable_count]


# ============================================================
# 8. LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


try:

    model = load_embedding_model()

except Exception as error:

    st.error(
        "Error loading embedding model."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# 9. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System Information")

    st.write(
        "**Vector Database:** FAISS"
    )

    st.write(
        "**Embedding Model:** "
        "all-MiniLM-L6-v2"
    )

    st.write(
        "**LLM:** Llama 3.2 3B"
    )

    st.write(
        "**Available Chunks:** "
        f"{len(chunks)}"
    )

    st.write(
        "**FAISS Vectors:** "
        f"{index.ntotal}"
    )

    st.divider()

    st.write(
        "Retrieval-Augmented Generation (RAG)"
    )

    st.write(
        "The system retrieves relevant information "
        "from the PDF before generating an answer."
    )


# ============================================================
# 10. SEARCH DOCUMENTS
# ============================================================

def search_documents(
    question,
    top_k=5
):

    # Create embedding for user question

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )


    # Convert to float32

    question_embedding = np.asarray(
        question_embedding,
        dtype=np.float32
    )


    # Make sure top_k does not exceed
    # number of available vectors

    actual_top_k = min(
        top_k,
        index.ntotal
    )


    # Search FAISS

    distances, indices = index.search(
        question_embedding,
        actual_top_k
    )


    results = []


    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx < 0:
            continue

        if idx >= len(chunks):
            continue

        results.append(
            {
                "text": chunks[idx],
                "distance": float(distance),
                "index": int(idx)
            }
        )


    return results


# ============================================================
# 11. GENERATE ANSWER USING LLAMA
# ============================================================

def generate_answer(
    question,
    results
):

    context_parts = []


    # Build context from retrieved chunks

    for i, result in enumerate(
        results,
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
    # RAG Prompt
    # --------------------------------------------------------

    prompt = f"""
You are a question-answering assistant for a
course technical document.

Answer the user's question using ONLY the
PDF context provided below.

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
    # Call Ollama
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
# 12. MAIN QUESTION INPUT
# ============================================================

st.subheader("🔎 Ask a Question")

question = st.text_input(
    "Enter your question:",
    placeholder="Example: What is Generative AI?"
)


# ============================================================
# 13. ASK BUTTON
# ============================================================

if st.button(
    "🔍 Ask Question",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        # ----------------------------------------------------
        # SEARCH PDF
        # ----------------------------------------------------

        with st.spinner(
            "Searching the PDF..."
        ):

            try:

                results = search_documents(
                    question,
                    top_k=5
                )

            except Exception as error:

                st.error(
                    "Error during document search."
                )

                st.code(
                    str(error)
                )

                st.stop()


        # ----------------------------------------------------
        # CHECK RESULTS
        # ----------------------------------------------------

        if not results:

            st.warning(
                "No relevant information was found."
            )

        else:

            st.success(
                f"Retrieved {len(results)} relevant PDF chunks."
            )


            # ------------------------------------------------
            # GENERATE ANSWER
            # ------------------------------------------------

            with st.spinner(
                "Generating answer using Llama 3.2..."
            ):

                try:

                    answer = generate_answer(
                        question,
                        results
                    )

                except Exception as error:

                    st.error(
                        "Error from Ollama."
                    )

                    st.code(
                        str(error)
                    )

                    st.stop()


            # ------------------------------------------------
            # DISPLAY ANSWER
            # ------------------------------------------------

            st.subheader(
                "🤖 AI Answer"
            )

            st.write(
                answer
            )


            # ------------------------------------------------
            # DISPLAY SOURCES
            # ------------------------------------------------

            st.subheader(
                "📖 Retrieved PDF Sources"
            )

            for i, result in enumerate(
                results,
                start=1
            ):

                with st.expander(
                    f"Retrieved Chunk {i}"
                ):

                    st.write(
                        result["text"]
                    )

                    st.caption(
                        f"FAISS vector index: "
                        f"{result['index']}"
                    )


# ============================================================
# 14. FOOTER
# ============================================================

st.divider()

st.caption(
    "Generative AI Course Project | "
    "Retrieval-Augmented Generation (RAG)"
)