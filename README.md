# Financial Document Intelligence & Q&A Chatbot

## Project Overview

This project is a Retrieval-Augmented Generation (RAG) application that answers questions about NVIDIA's financial reports and annual review documents.

It combines document loading, text chunking, semantic embeddings, vector similarity search, and a locally running Large Language Model (LLM) to generate answers grounded in retrieved document content.

## Objectives

* Load and process NVIDIA PDF reports.
* Split document text into smaller chunks while preserving page metadata.
* Generate embeddings for semantic search.
* Store and retrieve document chunks using FAISS.
* Generate context-based answers using Qwen through Ollama.
* Display source document names and page numbers.
* Evaluate the quality of answers to financial and general questions.

## Technology Stack

* Python
* LangChain
* PyPDF
* Hugging Face Sentence Transformers
* FAISS vector store
* Ollama
* Qwen 2.5 3B
* Regular expressions for selected financial-table normalization

## Architecture

NVIDIA PDF Reports → PDF Loading → Text Chunking → Embeddings → FAISS Vector Store → Similarity Retrieval → Context Formatting → Qwen LLM → Answer and Sources

## Project Structure

```text
financial-rag-chatbot/
├── data/
│   └── documents/          # Add PDF reports locally
├── vectorstore/
│   └── faiss_index/        # Generated vector index
├── src/
│   ├── __init__.py
│   ├── ingest.py
│   ├── retriever.py
│   └── chains.py
├── evaluation/
│   └── evaluate_rag.py
├── tests/
├── notebooks/
├── requirements.txt
├── .gitignore
└── README.md
```

## Prerequisites

* Python installed on your computer
* Ollama installed and running
* The Qwen model `qwen2.5:3b`
* NVIDIA PDF reports placed in `data/documents/`

## Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd financial-rag-chatbot
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository URL.

### 2. Create and activate a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Set up Ollama

Install Ollama from https://ollama.com/ if it is not already installed.

Download the model:

```bash
ollama pull qwen2.5:3b
```

Ensure Ollama is running before starting the chatbot.

### 5. Add PDF documents

Place your permitted NVIDIA PDF reports in:

```text
data/documents/
```

PDF reports are not included in this Git repository. Obtain them separately from an authorized source.

### 6. Build the vector store

From the project root, run:

```bash
python -m src.retriever
```

This command should build and save the FAISS index if the `src/retriever.py` entry point is configured to call `build_vectorstore()`.

### 7. Run the chatbot

```bash
python -m src.chains
```

Enter a question when prompted. The application generates an answer using retrieved document context and displays source information.

## Example Questions

* Where is NVIDIA headquartered?
* What are NVIDIA's main business segments?
* What products does NVIDIA develop?
* What was NVIDIA's Data Center revenue in fiscal year 2020?
* What was NVIDIA's revenue in fiscal year 2024?

## Evaluation

The project includes an evaluation script for testing financial questions and selected general questions.

Run:

```bash
python -m evaluation.evaluate_rag
```

Evaluation results depend on the documents, retrieval quality, prompt, model, and dependency versions. Record the actual results from your latest run rather than assuming a fixed accuracy.

## Current Limitations

* Retrieval can return irrelevant chunks alongside relevant information.
* Financial tables may require specialized parsing and year-to-value mapping.
* General questions can retrieve information from multiple report years.
* Answers must be checked against the cited source content.
* The project is an internship/portfolio implementation and should not be described as a commercial production deployment unless it has actually been deployed.

## Future Improvements

* Year-aware query routing.
* Hybrid keyword and vector retrieval.
* Reranking of retrieved chunks.
* Improved financial-table extraction.
* Citation and answer-grounding validation.
* LangGraph-based retrieval and answer-checking workflow.
* API and user-interface deployment.
* Automated retrieval and answer-quality evaluation.

## Acknowledgment

Developed as a document-question-answering and RAG learning project for financial report analysis.
