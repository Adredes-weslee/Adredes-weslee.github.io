---
layout: post
title: "A Deep Dive into Enterprise RAG: Design, Implementation, and Lessons Learned"
date: 2024-10-29 14:33:46 +0800 
categories: [ai, nlp, rag]
tags: [llms, retrieval-augmented-generation, vector-databases, langchain, python, system-design, mLOps]
author: Wes Lee
feature_image: /assets/images/2024-10-29-building-effective-rag-systems.jpg 
---

## Introduction: The RAG Revolution in Enterprise

Retrieval-Augmented Generation (RAG) is rapidly transforming how enterprises leverage Large Language Models (LLMs). By grounding LLMs with external, verifiable knowledge sources, RAG systems minimize hallucinations, provide up-to-date information, and enable domain-specific expertise. This post chronicles the journey of building and deploying a production-grade RAG system for technical documentation QA within an enterprise setting, detailing the technical choices, challenges, and solutions encountered.

> For a higher-level overview of this project, its strategic goals, the challenges it addresses, and its overall business impact, please see the [*Democratizing Enterprise Knowledge: The Custom RAG Engine Project*](/projects/rag-engine-project/) project page.

## Architecting the RAG Pipeline: Core Components

An effective RAG system is more than just an LLM and a vector database. It's a carefully orchestrated pipeline. Here’s a breakdown of the key components we engineered:

1.  **Document Processing & Chunking**: Converting diverse source documents into optimized, retrievable units.
2.  **Hybrid Embedding Strategy**: Generating meaningful vector representations for varied content types.
3.  **Vector Storage & Retrieval**: Efficiently indexing and searching embeddings.
4.  **Intelligent Query Processing**: Enhancing user queries and routing them effectively.
5.  **Contextual Assembly & Reranking**: Constructing the most relevant context for the LLM.
6.  **LLM Response Generation**: Prompting the LLM for accurate and coherent answers.
7.  **Evaluation & Self-Correction**: Continuously monitoring and improving performance.

Let's dive into the implementation details for each.

## Step 1: Document Processing - Beyond Naive Chunking

The foundation of any RAG system is how it ingests and prepares documents. Simply splitting text by token count is insufficient for enterprise content.

### From Fixed-Size to Semantic Chunking

We started with a common approach using `RecursiveCharacterTextSplitter` from LangChain:
```python
# Initial naive approach
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""] # Common separators
)
# chunks = text_splitter.split_text(document_text)
```
However, for diverse enterprise documents (markdown, code, PDFs), this led to suboptimal context. We evolved to a content-aware semantic chunking strategy.

### Implementing Content-Specific Chunkers

Different document types demand different chunking logic to preserve meaning. We developed specialized handlers:

```python
# In data_ingestion.py
from langchain.text_splitter import CharacterTextSplitter # Using CharacterTextSplitter for more control

def initialize_semantic_chunkers():
    """Initialize specialized text splitters for different document types."""
    # For markdown and general text, focusing on paragraphs and sections
    markdown_splitter = CharacterTextSplitter(
        separator="\n\n", # Split by double newlines first (paragraphs)
        chunk_size=512,   # Smaller chunk size for more focused context
        chunk_overlap=128, # Overlap to maintain context between chunks
        length_function=len,
        is_separator_regex=False,
    )
    
    # For code files, aiming to preserve function/class blocks if possible
    # This might involve more sophisticated parsing or simpler newline splitting for now
    code_splitter = CharacterTextSplitter(
        separator="\n", # More granular splitting for code
        chunk_size=256, # Smaller chunks for code snippets
        chunk_overlap=64,
        length_function=len,
        is_separator_regex=False,
    )
    return markdown_splitter, code_splitter

def extract_text_and_chunk_files(file_paths, markdown_splitter, code_splitter):
    """Process different file types with appropriate chunkers and add metadata."""
    all_chunks_with_metadata = []
    
    for file_path in file_paths:
        content = ""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        current_splitter = markdown_splitter
        file_type = "text"
        if file_path.endswith(('.py', '.ipynb', '.js', '.java', '.cpp')): # Add more extensions
            current_splitter = code_splitter
            file_type = "code"
        
        chunks = current_splitter.split_text(content)
        for chunk_content in chunks:
            all_chunks_with_metadata.append(
                {"text": chunk_content, "metadata": {"source": str(file_path), "type": file_type}}
            )
    return all_chunks_with_metadata
```
This ensures that code snippets aren't awkwardly split and markdown structure is respected. Metadata (like source file and type) is attached to each chunk for later use in filtering and citation.

## Step 2: Hybrid Embedding Strategy - Capturing Diverse Semantics

A one-size-fits-all embedding model struggles with the varied nature of enterprise data (technical docs vs. code). We implemented a hybrid embedding approach.

### Why Hybrid?
* **General Text Models** (e.g., `all-MiniLM-L6-v2`, `BAAI/bge-large-en`) excel at capturing semantic meaning in natural language.
* **Code-Specific Models** (e.g., `microsoft/graphcodebert-base`) are trained to understand code structure, variable names, and programming logic.

### Implementation
We load these models and generate embeddings separately for text and code chunks.

```python
# In model_loader.py / embedding_generation.py
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch # Ensure PyTorch is available

def load_embedding_models(device='cuda:0' if torch.cuda.is_available() else 'cpu'):
    """Load pre-trained models for text and code embeddings."""
    print(f"Loading models on device: {device}")
    # General text embedding model
    text_embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # Specialized code embedding model
    code_tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    code_embedding_model = AutoModel.from_pretrained("microsoft/graphcodebert-base").to(device)
    
    return text_embedding_model, code_tokenizer, code_embedding_model

def generate_text_embeddings_batch(texts, text_embedding_model, batch_size=32):
    """Generate embeddings for natural language text in batches."""
    return text_embedding_model.encode(texts, show_progress_bar=True, batch_size=batch_size)

def generate_code_embeddings_batch(code_snippets, code_tokenizer, code_embedding_model, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
    """Generate embeddings for code snippets in batches."""
    embeddings = []
    for i in range(0, len(code_snippets), 32): # Simple batching
        batch = code_snippets[i:i+32]
        inputs = code_tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            # Typically use the [CLS] token's embedding or mean pooling
            outputs = code_embedding_model(**inputs)
            # Example: Using pooler_output if available, or mean of last hidden state
            batch_embeddings = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
        embeddings.extend(batch_embeddings.cpu().numpy())
    return embeddings
```
These embeddings are then stored in separate FAISS vector stores, allowing for targeted retrieval.

## Step 3: Vector Storage & Advanced Retrieval

FAISS was chosen for its efficiency. We maintain separate indices for text and code.

```python
# In faiss_index.py
import faiss
import numpy as np

def create_and_populate_faiss_index(embeddings_list, dimension):
    """Create a FAISS index and add embeddings."""
    if not embeddings_list:
        return None
    embeddings_array = np.array(embeddings_list).astype('float32')
    index = faiss.IndexFlatL2(dimension) # L2 distance for similarity
    index.add(embeddings_array)
    return index

# Example usage after generating embeddings:
# code_faiss_index = create_and_populate_faiss_index(code_embeddings, code_embedding_dimension)
# text_faiss_index = create_and_populate_faiss_index(text_embeddings, text_embedding_dimension)

# faiss.write_index(code_faiss_index, 'faiss_code_index.idx')
# faiss.write_index(text_faiss_index, 'faiss_text_index.idx')
```

### Beyond Basic Similarity Search
Simple vector similarity search is often not enough. We implemented:

1.  **Hybrid Search (Dense + Sparse)**: Combining dense vector search (FAISS) with sparse retrieval (BM25) captures both semantic similarity and keyword relevance.
    ```python
    # In rag_chain.py, using LangChain's EnsembleRetriever
    from langchain.retrievers import EnsembleRetriever, BM25Retriever
    # Assuming `documents` is a list of LangChain Document objects
    # bm25_retriever = BM25Retriever.from_documents(documents)
    # faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    # ensemble_retriever = EnsembleRetriever(
    #     retrievers=[bm25_retriever, faiss_retriever],
    #     weights=[0.3, 0.7] # Weight sparse less than dense
    # )
    ```

2.  **Multi-Query Retrieval**: Generating multiple perspectives of the user's query to improve recall.
    ```python
    # Using LangChain's MultiQueryRetriever
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from langchain_community.llms import Ollama # Assuming Ollama for LLM

    # llm = Ollama(model="llama3:8b-instruct") # Or your chosen LLM
    # multi_query_retriever = MultiQueryRetriever.from_llm(
    #     retriever=faiss_retriever, # Your base retriever
    #     llm=llm
    # )
    ```

3.  **Self-Query Retrieval**: Allowing the LLM to write its own metadata filters based on the query.
    ```python
    # Using LangChain's SelfQueryRetriever
    from langchain.retrievers.self_query import SelfQueryRetriever
    from langchain.chains.query_constructor import AttributeInfo
    # metadata_field_info = [
    #     AttributeInfo(name="source", type="string", ...),
    #     AttributeInfo(name="type", type="string", ...)
    # ]
    # self_query_retriever = SelfQueryRetriever.from_llm(
    #     llm=llm,
    #     vectorstore=vector_store, # Your FAISS vector store
    #     document_contents="Technical documentation and code snippets",
    #     metadata_field_info=metadata_field_info,
    # )
    ```

## Step 4: Intelligent Query Processing & Routing

Not all questions are the same. We built a system to classify query types and route them to specialized handlers.

### Query Classification
An LLM call classifies the query (e.g., "code-related", "tabular/numerical", "general").

```python
# In question_handler.py
from langchain.prompts import ChatPromptTemplate
# llm = Ollama(model="llama3:8b-instruct") # Initialize your LLM

def determine_query_approach(user_question, llm_instance):
    """Determine the best processing approach for a given question."""
    template = """
Analyze the following question and determine if it's primarily:
1. About code, programming, or software implementation (respond with "CODE")
2. About data in tables, CSVs, or requires numerical analysis (respond with "TABLE")
3. A general question about concepts or text-based documentation (respond with "GENERAL")

Question: {query}
Classification (CODE, TABLE, or GENERAL):"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm_instance
    response = chain.invoke({"query": user_question}).content # Adjust based on LLM output
    
    if "CODE" in response.upper():
        return "code_handler"
    elif "TABLE" in response.upper():
        return "table_handler"
    else:
        return "general_handler"
```

### Handling Tabular Data
For "TABLE" queries, we used a Pandas DataFrame Agent (from LangChain) if CSVs or structured data were part of the knowledge base.

```python
# Simplified concept for tabular data handling
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# import pandas as pd

# def handle_tabular_query(question, df_list, llm_instance):
#     # Assuming df_list contains pandas DataFrames loaded from CSVs
#     # This requires more complex logic to select the right DataFrame or combine them
#     if df_list:
#         agent = create_pandas_dataframe_agent(llm_instance, df_list[0], verbose=True) # Simplified
#         try:
#             return agent.run(question)
#         except Exception as e:
#             print(f"Pandas agent error: {e}")
#     return "Could not process tabular query."
```

This routing ensures the right tools are used for the right job.

## Step 5: Context Assembly & Reranking - Crafting the Perfect Prompt

Retrieved documents must be assembled into a coherent context for the LLM.

### Reranking for Relevance
Retrieved chunks are reranked using a cross-encoder model to place the most relevant information at the top.

```python
# In rag_chain.py
from sentence_transformers import CrossEncoder

# reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_retrieved_documents(query, documents, top_n=5):
    """Reranks documents based on relevance to the query using a CrossEncoder."""
    if not documents:
        return []
    pairs = [[query, doc.page_content] for doc in documents] # Assuming LangChain Document objects
    scores = reranker_model.predict(pairs)
    
    scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:top_n]]
```

### Dynamic Context Assembly
We dynamically build the context, respecting the LLM's token limit. If a full document chunk is too large, we summarize it.

```python
# Simplified context assembly logic
# def count_tokens(text, tokenizer): # Implement token counting for your LLM
#     return len(tokenizer.encode(text))

# def summarize_document_content(content, llm_instance): # Implement summarization
#     # ...
#     pass

# def assemble_prompt_context(query, retrieved_docs, llm_instance, tokenizer, max_context_tokens=6000):
#     system_message = "You are an enterprise documentation assistant..."
#     ranked_docs = rerank_retrieved_documents(query, retrieved_docs) # Using the reranker
#     
#     context_str_parts = []
#     current_token_count = count_tokens(system_message, tokenizer)
#     
#     for doc in ranked_docs:
#         doc_content_tokens = count_tokens(doc.page_content, tokenizer)
#         source_info = f"Source: {doc.metadata.get('source', 'N/A')}\n"
#         source_tokens = count_tokens(source_info, tokenizer)
#         
#         if current_token_count + doc_content_tokens + source_tokens <= max_context_tokens:
#             context_str_parts.append(source_info + doc.page_content)
#             current_token_count += doc_content_tokens + source_tokens
#         else:
#             # Attempt to summarize if space allows (simplified)
#             summary_placeholder_tokens = 200 # Estimate for summary tokens
#             if current_token_count + summary_placeholder_tokens + source_tokens <= max_context_tokens:
#                 # summary = summarize_document_content(doc.page_content, llm_instance)
#                 # summary_tokens = count_tokens(summary, tokenizer)
#                 # if current_token_count + summary_tokens + source_tokens <= max_context_tokens:
#                 #     context_str_parts.append(source_info + "Summary: " + summary)
#                 #     current_token_count += summary_tokens + source_tokens
#             break # Stop if no more space
#     
#     final_context = "\n\n---\n\n".join(context_str_parts)
#     # Construct full prompt using system_message, final_context, and query
#     return full_prompt
```

## Step 6: LLM Response Generation & Self-Correction

We used a self-hosted Ollama instance with Llama 3.1 Instruct.

### Structured Prompting for QA
A clear prompt template guides the LLM.

```python
# In rag_chain.py
# qa_prompt_template = ChatPromptTemplate.from_messages([
#     ("system", """You are an enterprise documentation assistant...
#     Guidelines:
#     1. Answer ONLY based on the provided context.
#     2. If unsure, say "I don't have enough information...".
#     3. Include code examples from context when relevant.
#     4. Format with markdown."""),
#     ("human", "Context:\n{context}\n\nQuestion: {question}")
# ])

# chain = qa_prompt_template | llm # llm is your Ollama instance
# response = chain.invoke({"context": assembled_context, "question": user_query})
```

### Self-Correction Mechanism
The system can evaluate its own answer and trigger a fallback if the quality is low.

```python
# In evaluation_agent.py
# llm = Ollama(model="llama3:8b-instruct") # Or your chosen LLM

def evaluate_generated_answer(answer_text, original_question, llm_instance):
    """Evaluate answer quality using an LLM."""
    evaluation_prompt_text = f"""
Rate the quality of this answer on a scale from 1-10 based on relevance to the question, factual accuracy according to typical enterprise documentation, and completeness.
Question: {original_question}
Answer: {answer_text}
Provide only a numerical rating from 1-10. Rating:"""
    
    response = llm_instance.invoke(evaluation_prompt_text) # Assuming invoke returns a string
    try:
        score = float(response.strip())
        return score
    except ValueError:
        print(f"Could not parse evaluation score: {response}")
        return 5 # Default to a neutral score if parsing fails

# In rag_chain.py
# primary_answer = chain.invoke(...)
# quality_score = evaluate_generated_answer(primary_answer.content, user_query, llm)
# if quality_score < 7: # Threshold for fallback
#     # Trigger a fallback chain, perhaps with a modified prompt or different retriever settings
#     # fallback_answer = fallback_chain.invoke(...)
#     # final_answer = fallback_answer
# else:
#     # final_answer = primary_answer
```

## Step 7: Evaluation Framework - Measuring RAG Performance

We used the RAGAS framework for comprehensive evaluation.

```python
# Conceptual RAGAS evaluation setup
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# def run_ragas_evaluation(questions_list, generated_answers_list, retrieved_contexts_list_of_lists, ground_truths_list):
#     eval_data = {
#         "question": questions_list,
#         "answer": generated_answers_list,
#         "contexts": retrieved_contexts_list_of_lists, # List of lists of context strings
#         "ground_truth": ground_truths_list # Ground truth answers
#     }
#     eval_dataset = Dataset.from_dict(eval_data)
#     
#     result = evaluate(
#         dataset=eval_dataset,
#         metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
#     )
#     print(result)
#     return result
```
This allows tracking of faithfulness (hallucination), answer relevancy, and context quality.

## Deployment: Docker & Kubernetes

The system was containerized for secure and scalable enterprise deployment.

### Dockerization
A `Dockerfile` packages the RAG backend:
```dockerfile
# Backend service for RAG system (Simplified)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . . 
# Ensure models and FAISS indices are accessible, e.g., via volume mounts or copied in
# ENV MODEL_PATH="/app/models"
# ENV VECTOR_DB_PATH="/app/vector_stores"
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] # Assuming FastAPI in main.py
```

### Kubernetes for Scalability
`deployment.yaml` and `service.yaml` files manage the deployment on a Kubernetes cluster, ensuring scalability and resilience. This includes configurations for the RAG API, Ollama, and persistent volumes for models and vector stores.

## Performance Optimizations Implemented

### Vector Database Sharding (Conceptual)
For very large document sets, sharding the FAISS index can improve performance.
```python
# Conceptual sharding for FAISS (not fully implemented in provided snippets)
# class ShardedFAISSWrapper:
#     def __init__(self, embedding_function, num_shards=4):
#         self.num_shards = num_shards
#         self.shards = [FAISS(embedding_function=embedding_function, ...) for _ in range(num_shards)]
#         self.doc_to_shard_map = {}

#     def add_documents(self, documents):
#         for doc in documents:
#             shard_index = hash(doc.metadata.get("source", doc.page_content)) % self.num_shards
#             self.shards[shard_index].add_documents([doc])
#             self.doc_to_shard_map[doc.id] = shard_index # Assuming docs have IDs

#     def similarity_search(self, query, k=4):
#         results = []
#         for shard in self.shards: # Query all shards
#         _results.extend(shard.similarity_search(query, k=k))
#         # Rerank combined results
#         # ...
#         return sorted_results[:k]
```

### Response Caching
A Redis cache stores responses to frequently asked questions, reducing LLM load and latency.
```python
# Conceptual caching logic
# import redis
# import json
# import hashlib

# redis_client = redis.Redis(host='localhost', port=6379, db=0)
# CACHE_TTL_SECONDS = 3600 # 1 hour

# def get_query_cache_key(query_text):
#     return f"rag_cache:{hashlib.md5(query_text.lower().encode()).hexdigest()}"

# def get_cached_rag_response(query_text):
#     key = get_query_cache_key(query_text)
#     cached = redis_client.get(key)
#     return json.loads(cached) if cached else None

# def set_cached_rag_response(query_text, response_data):
#     key = get_query_cache_key(query_text)
#     redis_client.setex(key, CACHE_TTL_SECONDS, json.dumps(response_data))
```

## Key Lessons from the Enterprise Trenches

1.  **Document Quality is Paramount**: No RAG system can overcome poorly written, outdated, or ambiguous documentation. We implemented a document quality scoring heuristic to identify problematic content.
    ```python
    # import re
    # import textstat # Requires 'pip install textstat'

    # def calculate_document_quality_score(text_content):
    #     score = 0
    #     # Length bonus/penalty
    #     score += min(1, len(text_content) / 500) * 10 # Ideal length around 500 chars
    #     # Readability (Flesch Reading Ease)
    #     readability = textstat.flesch_reading_ease(text_content)
    #     if readability > 60: score += 10 # Good
    #     elif readability > 30: score += 5 # Fair
    #     # Presence of headers (simple check)
    #     if re.search(r'^#+\s', text_content, re.MULTILINE): score += 5
    #     # Presence of code blocks
    #     if "```" in text_content: score += 5
    #     return score
    ```
2.  **Understanding User Queries**: Classifying queries (factual, procedural, troubleshooting, etc.) helps tailor the RAG strategy and manage user expectations.
3.  **Continuous Feedback is Non-Negotiable**: A mechanism for users to rate answers and provide qualitative feedback is crucial for identifying weaknesses and guiding improvements. This data feeds back into evaluation datasets and fine-tuning efforts.

## Conclusion: The Evolving Landscape of Enterprise RAG

Building this RAG system was a journey of iterative refinement. Key successes include achieving secure, air-gapped LLM operation, handling diverse document formats effectively, and significantly improving access to technical knowledge. The modular design allows for ongoing enhancements, such as incorporating multi-modal data, deeper integration with structured databases, and more sophisticated agentic behaviors. RAG is not a static solution but an evolving capability that promises to unlock even greater value from enterprise data.

---

*To understand the broader strategic context, key features, and overall outcomes of this initiative, please visit the [project page](/projects/rag-engine-project/) project page. The source code is available on [GitHub](https://github.com/Adredes-weslee/Custom-RAG-Engine-for-Enterprise-Document-QA).*
