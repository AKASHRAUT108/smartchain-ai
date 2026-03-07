import os
import torch
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# ─── CONFIG ────────────────────────────────────────────────
FAISS_PATH  = "data/processed/faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL    = "deepset/roberta-base-squad2"

# ─── HELPER ────────────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ─── LOAD COMPONENTS ───────────────────────────────────────
def load_rag_chain():
    print("📦 Loading RAG components...")

    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name    = EMBED_MODEL,
        model_kwargs  = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True}
    )

    # Load FAISS index
    vector_store = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(
        search_type   = "similarity",
        search_kwargs = {"k": 3}
    )
    print("   ✅ Vector store loaded")

    # Load QA model directly (bypasses pipeline task registry)
    print(f"   🤖 Loading QA model: {QA_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
    model     = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL)
    model.eval()
    print("   ✅ QA model loaded — fully offline!\n")

    return retriever, tokenizer, model


def ask(retriever, tokenizer, model, question: str) -> dict:
    """
    Extractive QA — finds the exact answer span in retrieved context.
    No hallucination. No API. Works 100% offline.
    """
    print(f"\n❓ Question: {question}")

    # Step 1 — retrieve relevant chunks from FAISS
    source_docs = retriever.invoke(question)
    context     = format_docs(source_docs)

    # Step 2 — tokenize question + context
    inputs = tokenizer(
        question,
        context,
        return_tensors    = "pt",
        truncation        = True,
        max_length        = 512,
        padding           = True
    )

    # Step 3 — run model to get answer span positions
    with torch.no_grad():
        outputs     = model(**inputs)
        start_idx   = torch.argmax(outputs.start_logits)
        end_idx     = torch.argmax(outputs.end_logits) + 1

        # Calculate confidence score
        start_prob  = torch.softmax(outputs.start_logits, dim=1).max().item()
        end_prob    = torch.softmax(outputs.end_logits,   dim=1).max().item()
        confidence  = round((start_prob + end_prob) / 2, 3)

    # Step 4 — decode answer tokens back to text
    answer_tokens = inputs["input_ids"][0][start_idx:end_idx]
    answer        = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

    # Fallback if answer is empty
    if not answer or len(answer) < 2:
        answer = "Could not extract a specific answer. Please check the knowledge base."

    sources = [doc.page_content[:200] for doc in source_docs]

    print(f"💬 Answer:     {answer}")
    print(f"📊 Confidence: {confidence:.1%}")
    print(f"📎 Sources:    {len(sources)} chunks retrieved")

    return {
        "question":   question,
        "answer":     answer,
        "confidence": confidence,
        "sources":    sources
    }


if __name__ == "__main__":
    retriever, tokenizer, model = load_rag_chain()

    questions = [
        "What should I do if there is a port strike?",
        "How do I calculate safety stock?",
        "What is the recommended inventory level for A-items?",
        "How should I respond to a semiconductor shortage?",
        "What KPIs should I track for supply chain performance?",
        "What is the perfect order rate target?",
        "How much does air freight cost compared to ocean freight?"
    ]

    print("=" * 60)
    print("   SMARTCHAIN AI — RAG CHATBOT TEST")
    print("=" * 60)

    for q in questions:
        result = ask(retriever, tokenizer, model, q)
        print("-" * 60)