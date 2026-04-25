# Retrieval Augmented Generation

A comprehensive whitepaper and implementation guide on Retrieval-Augmented Generation (RAG).

**Authors:** Disha Bhavsar, Mayank Tiwari

## **Overview**

Retrieval-Augmented Generation (RAG) is an architectural pattern for Large Language Models (LLMs) designed to significantly enhance their capacity for generating responses that are both accurate and contextually pertinent. By grounding the generation process in external, verifiable knowledge sources, RAG effectively solves two major LLM limitations: **Knowledge Gaps** and **Hallucinations**.

This repository contains the whitepaper detailing the end-to-end architecture of RAG systems, along with a Python-based implementation of a complete RAG pipeline.

## **Architecture & Pipelines**

The whitepaper breaks down the RAG architecture into several distinct, modular pipelines:

### **1. Data Preparation Pipeline**

- **Document Ingestion:** Loading and parsing information from diverse sources (PDF, HTML, Word).
- **Chunking Strategies:** Detailed breakdown of Fixed-size, Recursive, Document-based, Semantic, Sentence-based, Token-based, Agentic, and Hierarchical chunking.
- **Embedding Generation:** Converting text into high-dimensional numerical vectors to capture semantic meaning.

### **2. Vector Database**

- Core concepts of storing, managing, and indexing high-dimensional vector data.
- **Indexing Methods:** Product Quantization (PQ) and Hierarchical Navigable Small World (HNSW) algorithms for fast Approximate Nearest Neighbor (ANN) searches.

### **3. Query Processing Pipeline**

- **Query Embedding:** Understanding TF-IDF, Word2Vec, and BERT for capturing query context.
- **Similarity Searches:** Utilizing Cosine, Dot product, Euclidean, and Manhattan distance metrics.

### **4. Context Construction Pipeline**

- **Context Packaging:** Concatenative and Hierarchical packaging to handle token limits and redundancy.
- **Prompt Template Construction:** Decomposing prompts into Instructions, Context, Delimiters, and Output Constraints.

### **5. Generation Pipeline**

- Mathematical breakdown of how Transformers predict the next token using the chain rule of probability over augmented contexts.

### **6. Post-Processing**

- Refining raw LLM output via formatting, post-hoc attribution (citations), confidence scoring, and safety/toxicity checks.

## **Applications and Advantages**

- **Reduced Hallucinations:** Responses are grounded in real, retrieved documents.
- **Up-to-date Responses:** Seamless integration of new knowledge without costly model fine-tuning.
- **Source Grounding:** Traceability and transparency through citations.
- **Domain Adaptation:** Reusable LLMs adaptable to various specialized domains via custom retrieval corpora.

## **Code Implementation**

The repository includes a bare-bones implementation of a RAG pipeline utilizing faiss, sentence-transformers, and Hugging Face's transformers.

### **Prerequisites**

```bash
pip install numpy faiss-cpu torch sentence-transformers transformers scikit-learn nltk
```

### **Quick Start**

The provided code (rag_pipeline.py) covers the complete lifecycle:

1. **Document Loading & Chunking:** Uses nltk for sentence tokenization and overlapping chunk generation.
2. **Indexing:** Uses SentenceTransformer("all-MiniLM-L6-v2") for embeddings and faiss.IndexFlatIP for vector storage.
3. **Retrieval & Reranking:** Implements cosine similarity-based retrieval and length-normalized reranking.
4. **Generation:** Uses the gpt2 model for answer generation based on a structured context prompt.

**Run the pipeline:**

```python
query = "What is RAG?"
answer = rag_pipeline(query, index, chunks)
print(answer)
```

## **Evaluation**

A robust RAG system requires continuous layered evaluation. Our guide covers:

- **Retrieval Evaluation:** Measuring Recall@K and Precision@K.
- **Retrieval Improvement:** Hybrid retrieval, query rewriting, and optimized chunking.
- **Generative Evaluation:** Assessing Faithfulness, Context Relevance, and Answer Correctness.
