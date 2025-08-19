
# Jupyter Notebook Specification: RAG Copilot for Enterprise Knowledge (On-Prem LLM)

## 1. Notebook Overview

**Learning Goals:**

*   Understand the end-to-end pipeline of a Retrieval-Augmented Generation (RAG) system for enterprise knowledge.
*   Learn how to ingest, chunk, embed, and index documents for efficient retrieval.
*   Implement different retrieval strategies, including dense, sparse, and hybrid approaches.
*   Design and evaluate prompt templates for RAG systems, incorporating citations and guardrails.
*   Evaluate the groundedness, hallucination, and answer quality of a RAG system.
*   Understand the basics of on-prem deployment considerations.

## 2. Code Requirements

**Expected Libraries:**

*   `transformers`: For pre-trained language models and tokenization.
*   `sentence-transformers`: For generating sentence embeddings.
*   `faiss-cpu`: For building and querying the vector index.
*   `nltk`: For text splitting.
*   `pypdf`: For reading PDF documents.
*   `requests`: For fetching HTML content.
*   `tqdm`: For progress bars.
*   `pandas`: For data manipulation and creating tables.
*   `matplotlib`: For generating plots.
*   `seaborn`: For improved plot aesthetics.
*   `evaluate`: For common evaluation metrics (e.g. rouge).

**Algorithms/Functions to Implement:**

*   `ingest_documents(file_paths)`: Reads documents from specified paths (PDF, HTML). Returns a list of document strings and their corresponding IDs.
*   `chunk_text(text, chunk_size, chunk_overlap)`: Splits text into smaller chunks based on the specified size and overlap.  Handles both semantic and standard splitting strategies.
*   `embed_text(text_chunks, model_name)`: Generates embeddings for a list of text chunks using a specified pre-trained embedding model.
*   `build_faiss_index(embeddings, dimension)`: Constructs a FAISS index from a set of embeddings.
*   `query_index(index, query_embedding, top_k)`: Performs a similarity search in the FAISS index and retrieves the top-k most relevant chunks.
*   `bm25_retrieval(query, documents, top_k)`: Performs BM25 retrieval on a list of documents for a given query and retrieves the top-k most relevant documents.
*   `hybrid_retrieval(query, dense_index, documents, top_k, alpha)`: Combines dense and sparse retrieval results based on a weighting factor (alpha).
*   `rerank_results(query, retrieved_chunks, model_name)`: Reranks the retrieved chunks using a cross-encoder model.
*   `create_rag_prompt(query, retrieved_chunks)`: Generates a prompt for the language model, incorporating the query and retrieved context.
*   `generate_answer(prompt, model_name)`: Generates an answer to the query using a pre-trained language model and the constructed prompt.
*   `evaluate_groundedness(answer, context)`: Evaluates how well the answer is supported by the retrieved context.
*   `evaluate_answer_quality(question, answer, model_name)`: Evaluates the quality of the answer using a pre-trained evaluation model (e.g., using ROUGE score).

**Visualizations:**

*   Bar chart comparing retrieval latency for different retrieval strategies (dense, sparse, hybrid).
*   Table showing example retrieved chunks and their corresponding relevance scores for different retrieval methods.
*   Scatter plot of answer quality vs. groundedness score.
*   Line graph of token generation speed during answer generation.
*   Histograms visualizing the distribution of chunk sizes for different chunking strategies.

## 3. Notebook Sections

1.  **Introduction: RAG Copilot for Enterprise Knowledge**

    *   Markdown Cell: Explain the concept of Retrieval-Augmented Generation (RAG) and its application in enterprise knowledge management. Describe the case study scenario of a compliance officer needing quick access to policy information. Define the learning objectives of the notebook. Explain that the notebook aims to provide a demonstrable RAG pipeline that is auditable.

2.  **Prerequisites: Tokens, Context Windows, and Costs**

    *   Markdown Cell:  Introduce fundamental concepts: tokens, context windows, cost, and latency. Explain the relationship between context window size and potential performance. Introduce the concept of tokenization: $ \text{Text} \rightarrow \text{Tokenizer} \rightarrow \text{Tokens} $. Describe the tradeoffs between model size, cost, and latency, i.e. bigger models have more parameters ($ \text{Parameters} \uparrow $), are more expensive to run ($ \text{Cost} \uparrow $), and have higher latency ($ \text{Latency} \uparrow $).
    *   Code Cell: Example code demonstrating tokenization using `transformers`.
    *   Markdown Cell: Explain the output of the tokenization code.

3.  **Data Ingestion and Preprocessing**

    *   Markdown Cell: Explain the data ingestion process: retrieving documents from various sources (PDF, HTML). Describe data normalization steps (e.g., removing special characters, converting to lowercase). Explain the importance of storing document IDs for citation purposes.
    *   Code Cell: Implementation of the `ingest_documents(file_paths)` function.
    *   Code Cell: Execution of the `ingest_documents` function with example file paths.
    *   Markdown Cell: Display the number of documents ingested.

4.  **Text Splitting and Semantic Chunking Strategies**

    *   Markdown Cell: Explain the importance of text splitting for RAG.  Introduce different chunking strategies: fixed-size chunking and semantic chunking. Explain the potential issues of simple chunking (e.g., splitting sentences). Explain different strategies for finding chunk boundaries.
    *   Code Cell: Implementation of the `chunk_text(text, chunk_size, chunk_overlap)` function.
    *   Code Cell: Execution of the `chunk_text` function with different chunk sizes and strategies.
    *   Markdown Cell: Display examples of the generated chunks.

5.  **Embeddings and Vector Indices**

    *   Markdown Cell: Introduce the concept of embeddings: converting text into numerical vectors. Explain the use of pre-trained sentence embedding models (e.g., Sentence Transformers). Introduce vector indices and their role in efficient similarity search. Explain the difference between approximate nearest neighbor (ANN) search and exact search.
    *   Code Cell: Implementation of the `embed_text(text_chunks, model_name)` function.
    *   Code Cell: Implementation of the `build_faiss_index(embeddings, dimension)` function.
    *   Code Cell: Execution of the `embed_text` and `build_faiss_index` functions.
    *   Markdown Cell: Show the shape of the embedding matrix and the size of the FAISS index.

6.  **Retrieval Strategies: Dense, Sparse (BM25), and Hybrid**

    *   Markdown Cell: Explain different retrieval strategies: dense retrieval (using vector index), sparse retrieval (using BM25), and hybrid retrieval.
        * Define dense retrieval as: $\text{argmax}_{\text{context} \in \text{Corpus}} \text{Similarity}(\text{Query Embedding}, \text{Context Embedding})$
        * Define BM25 as probabilistic retrieval function that calculates the probability a document is relevant to the query given the term frequencies of the document and the query: $ \text{BM25}(D,Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})} $ where $D$ is the document, $Q$ is the query, $q_i$ are the query terms, $f(q_i, D)$ is the term frequency of term $q_i$ in document $D$, $|D|$ is the length of the document $D$, $\text{avgdl}$ is the average document length in the corpus, $k_1$ and $b$ are tuning parameters.
    *   Code Cell: Implementation of the `query_index(index, query_embedding, top_k)` function.
    *   Code Cell: Implementation of the `bm25_retrieval(query, documents, top_k)` function.
    *   Code Cell: Implementation of the `hybrid_retrieval(query, dense_index, documents, top_k, alpha)` function.
    *   Code Cell: Execution of the retrieval functions with example queries and parameter settings.
    *   Markdown Cell: Display the retrieved chunks and their corresponding relevance scores for each retrieval method.

7.  **Reranking (Optional)**

    *   Markdown Cell: Explain the concept of reranking and its benefits for improving retrieval accuracy. Explain how reranking uses cross-encoders to model the query-document interaction.
    *   Code Cell: Implementation of the `rerank_results(query, retrieved_chunks, model_name)` function.
    *   Code Cell: Execution of the `rerank_results` function with example queries and retrieved chunks.
    *   Markdown Cell: Display the reranked chunks and their updated relevance scores.

8.  **Prompt Templating for RAG**

    *   Markdown Cell: Explain the importance of prompt engineering for RAG systems. Describe different prompt templates for incorporating retrieved context and citations. Discuss the use of system instructions and guardrails in prompts.
    *   Code Cell: Implementation of the `create_rag_prompt(query, retrieved_chunks)` function.
    *   Code Cell: Execution of the `create_rag_prompt` function with example queries and retrieved chunks.
    *   Markdown Cell: Display the generated prompt.

9.  **Answer Generation**

    *   Markdown Cell: Explain the process of generating an answer using a pre-trained language model and the constructed prompt. Discuss the use of different decoding strategies (e.g., greedy decoding, beam search).
    *   Code Cell: Implementation of the `generate_answer(prompt, model_name)` function.
    *   Code Cell: Execution of the `generate_answer` function with the generated prompt.
    *   Markdown Cell: Display the generated answer.

10. **Evaluation: Groundedness, Hallucination Detection, and Answer Quality**

    *   Markdown Cell: Explain the key metrics for evaluating RAG systems: groundedness, hallucination detection, and answer quality. Define groundedness as the degree to which the claims made in the answer can be supported by the retrieved context. Hallucination is the act of the LLM generating information that it was not given from the prompt. Answer quality can be measured by comparing the answer to a reference answer using metrics like ROUGE.
    *   Code Cell: Implementation of the `evaluate_groundedness(answer, context)` function.
    *   Code Cell: Implementation of the `evaluate_answer_quality(question, answer, model_name)` function.
    *   Code Cell: Execution of the evaluation functions with example questions, answers, and context.
    *   Markdown Cell: Display the evaluation results.

11. **Visualization of Evaluation Results**

    *   Markdown Cell: Introduce the visualizations that will be used to analyze the evaluation results.
    *   Code Cell: Generate a scatter plot of answer quality vs. groundedness score.
    *   Code Cell: Generate a bar chart comparing retrieval latency for different retrieval strategies.
    *   Markdown Cell: Explain the insights gained from the visualizations.

12. **Security Considerations (Brief)**

    *   Markdown Cell: Briefly discuss security considerations for on-prem RAG systems.  Mention topics like PII redaction, input sanitization, and prompt injection prevention. Do not implement any actual security measures, just highlight their importance.

13. **Conclusion**

    *   Markdown Cell: Summarize the key concepts covered in the notebook. Emphasize the importance of RAG for enterprise knowledge management. Suggest further exploration of topics like on-prem deployment, scaling, and continuous evaluation.

