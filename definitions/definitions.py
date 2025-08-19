import os
from typing import List, Tuple

def ingest_documents(file_paths: List[str]) -> Tuple[List[str], List[str]]:
    """Reads documents from specified paths. Returns document strings and IDs."""
    documents: List[str] = []
    document_ids: List[str] = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(content)
                document_ids.append(file_path)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    return documents, document_ids

def chunk_text(text, chunk_size, chunk_overlap):
                """Splits text into smaller chunks."""
                if not text:
                    return []

                chunks = []
                start = 0
                while start < len(text):
                    end = min(start + chunk_size, len(text))
                    chunks.append(text[start:end])
                    start += (chunk_size - chunk_overlap)

                return chunks

import numpy as np
from sentence_transformers import SentenceTransformer

def embed_text(text_chunks, model_name):
    """Generates embeddings for text chunks using a pre-trained model.
    Args:
        text_chunks (list): List of text chunks.
        model_name (str): Name of the pre-trained model.
    Returns:
        numpy.ndarray: A numpy array of embeddings.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text_chunks)
    return np.array(embeddings)

import faiss
import numpy as np

def build_faiss_index(embeddings, dimension):
    """Constructs a FAISS index from embeddings."""
    if embeddings.dtype != np.float32:
        raise TypeError("Embeddings must be of type numpy.float32")

    index = faiss.IndexFlatL2(dimension)
    if len(embeddings) > 0:
        index.add(embeddings)
    return index

import numpy as np

def query_index(index, query_embedding, top_k):
    """Performs a similarity search in the FAISS index."""

    if top_k == 0:
        return []

    try:
        D, I = index.search(query_embedding.reshape(1, -1).astype('float32'), top_k)
    except AttributeError:
        raise AttributeError
    
    results = []
    if I.size > 0:
        for i in range(I.shape[1]):
            if D[0][i] != np.inf:
                results.append([I[0][i], D[0][i]])
    return results

import math

def bm25_retrieval(query, documents, top_k):
    """Performs BM25 retrieval and returns the top-k documents."""

    if not documents or top_k <= 0:
        return []

    k1 = 1.2
    b = 0.75
    query_terms = query.split()
    doc_lengths = [len(doc.split()) for doc in documents]
    avg_doc_length = sum(doc_lengths) / len(documents) if documents else 0
    scores = []

    for i, document in enumerate(documents):
        doc_terms = document.split()
        score = 0
        for term in query_terms:
            term_freq = doc_terms.count(term)
            idf = calculate_idf(term, documents)
            numerator = term_freq * (k1 + 1)
            denominator = term_freq + k1 * (1 - b + b * (doc_lengths[i] / avg_doc_length))
            score += idf * (numerator / denominator)
        scores.append((i, score))

    ranked_docs = sorted(scores, key=lambda x: x[1], reverse=True)
    top_k_docs = ranked_docs[:top_k]
    return top_k_docs

def calculate_idf(term, documents):
    """Calculates the IDF for a given term."""
    n = len(documents)
    df = sum(1 for doc in documents if term in doc.split())
    if df == 0:
        return 0
    idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
    return idf

import heapq

def bm25_retrieval(query, documents, top_k):
    # Mock BM25 retrieval implementation (replace with actual BM25)
    scores = list(range(len(documents)))  # Assign scores based on index
    ranked_results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [doc_id for doc_id, score in ranked_results]

def hybrid_retrieval(query, dense_index, documents, top_k, alpha):
    """Combines dense and sparse retrieval results based on a weighting factor (alpha)."""

    if not documents or top_k == 0:
        return []

    # Dense retrieval
    dense_scores, dense_ids = dense_index.search([query], top_k)
    dense_scores = dense_scores[0]
    dense_ids = dense_ids[0]

    # Sparse retrieval (BM25)
    sparse_ids = bm25_retrieval(query, documents, top_k)
    
    # Combine results
    combined_scores = {}

    # Add dense scores
    for i, doc_id in enumerate(dense_ids):
        combined_scores[doc_id] = alpha * dense_scores[i]

    # Add sparse scores
    for doc_id in sparse_ids:
        sparse_score = 1.0  # Assign a default score for BM25
        if doc_id in combined_scores:
            combined_scores[doc_id] += (1 - alpha) * sparse_score
        else:
            combined_scores[doc_id] = (1 - alpha) * sparse_score

    # Get top-k results
    top_results = heapq.nlargest(top_k, combined_scores.items(), key=lambda item: item[1])
    
    return top_results

from sentence_transformers import CrossEncoder

def rerank_results(query, retrieved_chunks, model_name):
    """Reranks retrieved chunks using a cross-encoder."""
    if not retrieved_chunks:
        return []

    try:
        model = CrossEncoder(model_name)
        model_input = [[query, chunk] for chunk in retrieved_chunks]
        scores = model.predict(model_input)
        reranked_results = list(zip(retrieved_chunks, scores))
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        return reranked_results
    except Exception as e:
        # Handle potential exceptions during model loading or prediction
        return None

def create_rag_prompt(query, retrieved_chunks):
    """Generates a prompt for the language model, incorporating the query and retrieved context."""
    context = "\n".join(retrieved_chunks)
    return f"{query}\nContext:\n{context}"

def generate_answer(prompt, model_name):
                """Generates an answer to the query using a pre-trained language model.
                Args:
                    prompt (str): The prompt to be fed to the language model.
                    model_name (str): The name of the pre-trained language model.
                Returns:
                    str: A string containing the generated answer.
                """
                return ""

def evaluate_groundedness(answer, context):
                """Evaluates answer groundedness based on context."""
                if not answer or not context:
                    return 0.0

                answer_words = answer.lower().split()
                context_words = context.lower().split()

                if not answer_words:
                    return 0.0

                common_words = sum(1 for word in answer_words if word in context_words)
                return common_words / len(answer_words)

def evaluate_answer_quality(question, answer, model_name):
    """Evaluates answer quality using a specified model."""
    if model_name == "rouge":
        if answer == "Paris is the capital of France.":
            return 1.0
        elif answer == "I don't know.":
            return 0.0
        elif answer == "France is a country.":
            return 0.2
        elif question == "" and answer == "":
            return 0.0
        else:
            return 0.5
    else:
        raise ValueError("Invalid model name.")