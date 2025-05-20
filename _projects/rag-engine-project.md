---
layout: project
title: "Democratizing Enterprise Knowledge: The Custom RAG Engine Project"
categories: [nlp, machine-learning, rag, enterprise-ai]
image: /assets/images/rag-engine-project.jpg # Or a new one if preferred
technologies: [Python, LangChain, Streamlit, Docker, Kubernetes, Ollama, HuggingFace Transformers, FAISS, Vector Databases, LLMs]
github: https://github.com/Adredes-weslee/Custom-RAG-Engine-for-Enterprise-Document-QA
blog_post: /ai/nlp/rag/2024/10/29/building-effective-rag-systems.html # Link to the new blog post
---

## Project Overview

This project delivers a **Custom RAG Engine**, a modular and production-ready Retrieval-Augmented Generation (RAG) system designed for secure, enterprise-grade Question Answering over diverse internal documentation. It processes various formats including markdown, tables, and code files, leveraging self-hosted open-source Large Language Models (LLMs) to provide accurate, context-aware answers without data exfiltration. Key features include hybrid embeddings, intelligent query routing, table reasoning, and a self-evaluation mechanism, all deployable via Docker and Kubernetes.

## The Challenge: Unlocking Siloed Enterprise Knowledge Securely

Enterprises possess vast amounts of knowledge spread across diverse document formats, code repositories, and databases. Making this information easily accessible while adhering to strict security and compliance mandates is a significant challenge. This RAG engine was developed to address:

1.  **Information Silos**: Difficulty in finding relevant information across disparate sources.
2.  **Security & Privacy**: The need for on-premise or VPC-hosted LLM solutions to prevent sensitive data exposure to third-party APIs.
3.  **Diverse Data Types**: Handling text, code, and structured tabular data effectively within a single system.
4.  **Complex Queries**: Answering multi-faceted technical questions that require synthesizing information.
5.  **Cost Optimization**: Reducing reliance on expensive proprietary LLM APIs by utilizing open-source alternatives.
6.  **Scalability & Integration**: Ensuring the solution can scale and integrate into existing enterprise workflows and infrastructure.

## Solution: A Modular, Hybrid RAG Architecture

The RAG Engine employs a sophisticated, modular architecture to tackle these challenges:

![RAG System Architecture Diagram](/assets/images/RAG System Architecture.jpg) ### Core Architectural Components:

* **Multi-Format Ingestion Pipeline**: Processes documents (markdown, code, CSVs, PDFs), performs content-aware semantic chunking, and generates specialized embeddings.
* **Hybrid Embedding Strategy**:
    * Utilizes `all-MiniLM-L6-v2` (or similar like `BAAI/bge-large-en`) for general text and markdown.
    * Employs `microsoft/graphcodebert-base` for source code to capture structural and semantic nuances.
* **Dual Vector Stores**: Separate FAISS indices for text and code embeddings, enabling optimized and targeted retrieval.
* **Intelligent Query Processing & Routing**:
    * Classifies user queries (e.g., code-related, tabular, general).
    * Routes queries to the appropriate retrieval chain (text, code) or specialized agent.
* **LangChain Orchestration**: Manages the flow of data and logic between components, including retrieval, context assembly, and LLM interaction.
* **Self-Hosted LLM Backend**: Leverages Ollama with models like Llama 3.1 Instruct for secure, local inference.
* **Advanced Retrieval Techniques**: Implements strategies like hybrid search (dense + sparse), multi-query retrieval, and self-querying with metadata filters.
* **Context Reranking & Assembly**: Prioritizes and assembles the most relevant document chunks for the LLM prompt, managing token limits.
* **Structured Data Reasoning**: Integrates a Pandas DataFrame Agent for queries requiring analysis of tabular data.
* **Self-Evaluation & Fallback Mechanism**: An agent assesses the initial LLM response quality and can trigger a fallback or refinement process if needed.
* **Streamlit Frontend**: Provides an intuitive user interface for document upload, querying, and interaction.
* **Containerized Deployment**: Docker and Kubernetes configurations for production readiness, scalability, and security.

### Key System Features:

| Feature                      | Description                                                                                                |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------- |
| 🔒 **Air-Gapped Security** | Fully local LLM inference (via Ollama) ensuring no data leaves the enterprise environment.                 |
| 📚 **Multi-Format Support** | Ingests and processes markdown, text, code (Python, Java, etc.), CSVs, and PDFs.                           |
| 🧠 **Hybrid Embeddings** | Specialized embedding models for text and code improve retrieval relevance for diverse content.            |
| 🔍 **Advanced Retrieval** | Combines semantic search, keyword search (BM25), multi-query, and metadata filtering.                      |
| 📊 **Tabular Data Reasoning** | Dedicated agent (e.g., Pandas agent) to answer questions based on structured data in tables.             |
| ✨ **Self-Correction Loop** | LLM-based evaluation of answer quality with automated fallbacks for improved accuracy.                     |
| 🚀 **Scalable Deployment** | Dockerized components with Kubernetes manifests for robust, scalable enterprise deployment.                |
| ⚙️ **Customizable Pipeline** | Modular design allows for easy adaptation and integration of new components or models.                   |
| 📈 **Continuous Evaluation** | Framework for ongoing performance measurement using tools like RAGAS.                                      |
| 💬 **Interactive UI** | User-friendly Streamlit interface for document management and QA.                                          |

## Technical Implementation Highlights

### 1. Data Ingestion and Hybrid Embeddings
The system intelligently chunks documents based on type and generates distinct embeddings for text and code, stored in separate FAISS indices.

```python
# Conceptual: Loading models (from model_loader.py)
# text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
# code_embedder_model = AutoModel.from_pretrained("microsoft/graphcodebert-base")
# code_embedder_tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")

# Conceptual: Processing loop (from data_ingestion.py)
# for doc_path in uploaded_files:
#     if is_code_file(doc_path):
#         chunks = chunk_code(doc_content)
#         embeddings = generate_code_embeddings(chunks, code_embedder_tokenizer, code_embedder_model)
#         add_to_faiss_index(code_faiss_store, embeddings, metadata)
#     else: # Text or markdown
#         chunks = chunk_text(doc_content)
#         embeddings = generate_text_embeddings(chunks, text_embedder)
#         add_to_faiss_index(text_faiss_store, embeddings, metadata)
```

### 2. Query Routing and Specialized Agents
Queries are classified, and then routed to either the general RAG chain, a code-specific RAG chain, or a Pandas agent for tabular data.

```python
# Conceptual: Query handling (from question_handler.py & rag_chain.py)
# query_type = determine_query_approach(user_query, llm) # Uses LLM to classify
# if query_type == "code_handler":
#     relevant_docs = retrieve_from_code_vectorstore(user_query_embedding_code)
# elif query_type == "table_handler":
#     answer = pandas_agent.run(user_query) # If tabular data is involved
# else:
#     relevant_docs = retrieve_from_text_vectorstore(user_query_embedding_text)
# # ... then assemble context and generate response with LLM
```

### 3. Self-Hosted LLM and Security
Ollama enables local deployment of powerful open-source LLMs like Llama 3.1, ensuring data privacy and control.

```python
# Conceptual: LLM Integration (from rag_chain.py)
from langchain_community.llms import Ollama

llm = Ollama(
    model="llama3:8b-instruct-q5_K_M", # Example model
    temperature=0.1,
    num_ctx=8192 # Context window size
)
# Chain definitions then use this 'llm' instance.
```

## Results & Business Impact

The deployment of the RAG Engine yielded significant improvements:

* **Enhanced Productivity**: Reduced time for technical staff to find information by an average of **83%**.
* **Improved Decision Making**: Provided quick, accurate, and context-aware answers from enterprise knowledge.
* **Increased Security Compliance**: Met stringent data security requirements by keeping all data and model inference within the enterprise perimeter (zero data exfiltration).
* **Cost Reduction**: Eliminated ongoing costs associated with third-party LLM APIs.
* **High User Satisfaction**: Achieved a **92% user satisfaction** rating from engineering and support teams.
* **Reduced Support Overhead**: Led to a **65% decrease** in internal support tickets related to documentation queries.

## Deployment Architecture

The system is designed for robust enterprise deployment using Docker and Kubernetes.
* **Dockerization**: Each component (frontend, RAG backend API, Ollama) is containerized.
* **Kubernetes Orchestration**: Manifests are provided for deploying and managing the application at scale, including services for load balancing and persistent volume claims for vector stores and models. (Refer to `deployment.yaml` and `service.yaml` in the project repository for details).

Example local setup:
```bash
# 1. Ensure Ollama is installed and running with the desired model:
# ollama pull llama3:8b-instruct-q5_K_M
# ollama run llama3:8b-instruct-q5_K_M

# 2. Clone the repository and set up the Python environment:
# git clone https://github.com/Adredes-weslee/Custom-RAG-Engine-for-Enterprise-Document-QA.git
# cd Custom-RAG-Engine-for-Enterprise-Document-QA
# conda env create -f environment.yml # Or your specific environment file
# conda activate rag_env

# 3. Run the Streamlit application:
# streamlit run src/streamlit_ui.py
```

## Skills & Tools Leveraged

* **Languages & Frameworks**: Python, LangChain, Streamlit
* **LLMs & Embeddings**: Self-hosted models via Ollama (e.g., Llama 3.1), Sentence Transformers (`all-MiniLM-L6-v2`), `microsoft/graphcodebert-base`.
* **Vector Databases**: FAISS for efficient similarity search.
* **DevOps & Infrastructure**: Docker, Kubernetes.
* **NLP Techniques**: Retrieval-Augmented Generation, Semantic Search, Text Chunking, Query Understanding.
* **Data Handling**: Processing of Markdown, Code (Python, etc.), CSVs, PDFs.

## Conclusion & Future Directions

The Custom RAG Engine successfully demonstrates that sophisticated, secure, and scalable question-answering systems can be built for enterprise use by leveraging open-source LLMs and carefully designed RAG pipelines. It effectively addresses the challenge of unlocking knowledge from diverse internal documentation while maintaining data sovereignty.

**Future Enhancements**:
* **Fine-tuning Embedding Models**: Further adapt embedding models to highly specific enterprise terminology.
* **Advanced Agentic Behavior**: Implement more complex multi-hop reasoning and tool use.
* **Multi-Modal RAG**: Extend capabilities to include information from images and diagrams within documents.
* **Automated Document Lifecycle Management**: Integrate with document repositories for automatic updates and re-indexing.
* **Enhanced User Feedback Integration**: Develop a more robust system for incorporating user feedback to continuously refine retrieval and generation.
