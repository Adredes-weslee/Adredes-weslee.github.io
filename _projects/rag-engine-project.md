---
layout: project
title: "Custom RAG Engine for Enterprise Document QA"
categories: nlp machine-learning rag
image: /assets/images/placeholder.svg
technologies: [Python, LangChain, Streamlit, Docker, Kubernetes, Ollama, HuggingFace]
github: https://github.com/Adredes-weslee/Custom-RAG-Engine-for-Enterprise-Document-QA
---

## Project Overview

Designed and deployed a modular, containerized Retrieval-Augmented Generation (RAG) system to support structured QA over enterprise documentation — including markdowns, tables, and code files — using self-hosted open-source LLMs.

## Architecture

![RAG System Architecture](/assets/images/rag-architecture.png)

## Key Components

- **Frontend**: Streamlit UI with file uploader and chat interface
- **Backend**: Ollama-hosted LLaMA 3.1 Instruct model for local inference
- **Dual Embeddings**: 
  - MiniLM for text content
  - GraphCodeBERT for code search
- **LangChain Orchestration**: Smart routing between vector retriever, pandas agent for table reasoning, and fallback evaluators
- **Self-Reflection Loop**: Quality-checking agents for response validation and rerouting
- **Deployment-Ready**: Dockerized app with Kubernetes manifests for backend, frontend, and Ollama container integration

## Multi-Format Processing

The system supports multiple document types:
- Markdown files
- CSV/tabular data
- Python source code
- PDF documents

Each document type receives specialized preprocessing, chunking, and metadata tagging to optimize retrieval.

## Technical Implementation

```python
# Sample code showing the dual embedding pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Text embeddings for general content
text_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Specialized embeddings for code
code_embeddings = HuggingFaceEmbeddings(
    model_name="microsoft/graphcodebert-base",
    model_kwargs={'device': 'cpu'}
)

# Create vector stores with different embedding models
text_vectordb = Chroma(
    collection_name="text_documents",
    embedding_function=text_embeddings
)

code_vectordb = Chroma(
    collection_name="code_documents",
    embedding_function=code_embeddings
)
```

## Results and Impact

This project delivered a secure, extensible RAG platform for document QA that is:
- Fully air-gapped for enterprise security
- Kubernetes-ready for scalable deployment
- Designed for structured enterprise use cases
- Capable of handling complex multi-format document collections

## Technologies Used

- **Python** - Core programming language
- **LangChain** - For RAG orchestration
- **Streamlit** - User interface
- **FastAPI** - Backend service
- **Docker** - Containerization
- **Kubernetes** - Deployment orchestration
- **Ollama** - Local LLM hosting
- **HuggingFace Transformers** - Embedding models
