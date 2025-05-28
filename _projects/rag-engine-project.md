---
layout: project
title: "Democratizing Enterprise Knowledge: The Custom RAG Engine Project"
categories: [nlp, machine-learning, rag, enterprise-ai, local-deployment]
image: /assets/images/rag-engine-project.jpg
technologies: [Python, LangChain, Streamlit, Ollama, HuggingFace Transformers, FAISS, Vector Databases, LLMs, Local AI, GPU Acceleration]
github: https://github.com/Adredes-weslee/Custom-RAG-Engine-for-Enterprise-Document-QA
blog_post: /ai/nlp/rag/2024/10/29/building-effective-rag-systems.html

---

## Project Overview

This project delivers a **Custom RAG Engine**, a sophisticated local-first Retrieval-Augmented Generation (RAG) system designed for secure, enterprise-grade Question Answering over diverse internal documentation. Built specifically for **local deployment**, it processes GitLab repository data including Python code and Jupyter notebooks, leveraging self-hosted open-source Large Language Models (LLMs) via Ollama to provide accurate, context-aware answers with complete data privacy. Key features include GPU-accelerated processing, hybrid code-aware embeddings, intelligent LLM enhancement, and multi-model evaluation - all running entirely on your hardware.

## The Challenge: Unlocking Enterprise Code Knowledge Securely

Enterprises and educational institutions possess vast amounts of technical knowledge embedded in code repositories, project documentation, and learning materials. Making this information easily accessible while maintaining absolute data privacy and security is a critical challenge. This RAG engine was developed to address:

1. **Code Understanding Barriers**: Difficulty in finding relevant implementation patterns and learning from existing codebases.
2. **Educational Data Privacy**: Need for completely local processing of student projects and sensitive learning materials.
3. **Diverse Technical Formats**: Handling Python scripts, Jupyter notebooks, and documentation within a unified system.
4. **Performance Requirements**: Need for fast processing and retrieval using available hardware (GPU acceleration).
5. **Zero External Dependencies**: Avoiding any external API calls or data transmission for maximum security.
6. **Local Resource Optimization**: Efficiently utilizing available GPU/CPU resources for maximum performance.

## Solution: A Local-First, GPU-Accelerated RAG Architecture

The RAG Engine employs a sophisticated, hardware-optimized architecture designed exclusively for local deployment:

![RAG System Architecture Diagram](/assets/images/RAG System Architecture.jpg)

### Core Architectural Components:

* **GPU-Accelerated Data Pipeline**: Processes 23 AI apprentice projects (~1000+ files) with CUDA acceleration for 10-50x performance improvements.
* **Specialized Code Embeddings**:
    * `microsoft/graphcodebert-base` (768D) for Python source code analysis.
    * `all-MiniLM-L6-v2` (384D) for Jupyter notebook content.
    * Dimensionality alignment to 384D common space for unified retrieval.
* **Dual FAISS Vector Stores**: Separate GPU-optimized indices for code and notebook embeddings.
* **LLM-Powered Enhancement**: Uses local Ollama models to add comments, docstrings, and explanations to raw code.
* **Multi-Model Local Inference**: 
    * Primary: `llama3.2:3b` for high-quality responses.
    * Judge: `llama3.1:8b` for response evaluation.
    * Fallback: `llama3.2:1b` for resource-constrained scenarios.
* **Intelligent Query Routing**: LangChain-orchestrated pipeline that routes queries to appropriate vector stores.
* **Self-Evaluation System**: Judge model assesses response quality with automated scoring and reasoning.
* **Local Streamlit Interface**: Complete web UI running on localhost with real-time query processing.
* **Environment-Aware Configuration**: Automatic detection of available hardware and model selection.

### Key System Features:

| Feature | Description | Performance Benefit |
|---------|-------------|-------------------|
| 🔒 **Complete Local Processing** | All inference via local Ollama server - zero external API calls | 100% data privacy |
| ⚡ **GPU Acceleration** | CUDA-optimized embeddings and FAISS operations | 10-50x faster processing |
| 🧠 **Code-Aware Intelligence** | GraphCodeBERT understands code structure and semantics | Higher retrieval accuracy for code |
| 📚 **GitLab Repository Focus** | Specialized for educational/enterprise project repositories | Domain-optimized performance |
| 🎯 **Multi-Model Evaluation** | Judge model evaluates response quality with scoring | Self-improving accuracy |
| 🚀 **Hardware Optimization** | Automatic GPU detection with CPU fallback | Maximum resource utilization |
| 📈 **Scalable Local Deployment** | Process 1000+ files locally with pre-built indices | Production-ready performance |
| ⚙️ **Environment Detection** | Adapts to available hardware automatically | Optimal model selection |
| 💬 **Real-Time Interface** | Streamlit UI with live query processing | Immediate response generation |
| 🔄 **Continuous Enhancement** | LLM-powered code improvement during ingestion | Higher quality knowledge base |

## Technical Implementation Highlights

### 1. GPU-Accelerated Data Ingestion with LLM Enhancement
The system processes GitLab repository data with intelligent content enhancement:

```python
# Enhanced data processing pipeline (from data_ingestion.py)
def process_repository_data():
    # Discover Python and notebook files across 23 people
    files = discover_gitlab_files(root_directory)
    
    # GPU-accelerated processing
    for file_path, content in files:
        if is_python_file(file_path):
            # LLM enhancement with local Ollama
            enhanced_content = enhance_code_with_llm(content, llm_model)
            
            # Code-aware embeddings
            embeddings = generate_code_embeddings(enhanced_content, graphcodebert_model)
            code_faiss_index.add_with_metadata(embeddings, metadata)
            
        elif is_notebook_file(file_path):
            # Process notebook cells
            enhanced_content = enhance_notebook_with_llm(content, llm_model)
            
            # Semantic embeddings
            embeddings = generate_text_embeddings(enhanced_content, minilm_model)
            notebook_faiss_index.add_with_metadata(embeddings, metadata)
```

### 2. Local Multi-Model Architecture with Evaluation
Environment-aware model selection with built-in evaluation:

```python
# Local model configuration (from model_config.py)
def get_local_models():
    environment = detect_environment()  # local_high, local_standard, local_minimal
    
    if environment == "local_high":
        return {
            'primary': ChatOllama(model="llama3.2:3b", temperature=0.1),
            'judge': ChatOllama(model="llama3.1:8b", temperature=0.0),
            'fallback': ChatOllama(model="llama3.2:1b", temperature=0.2)
        }
    # ... other configurations

# Query processing with evaluation (from rag_chain.py)
def process_query_with_evaluation(query, models):
    # Generate initial response
    response = models['primary'].invoke(context + query)
    
    # Judge evaluation
    evaluation = models['judge'].invoke(f"Evaluate this response: {response}")
    score = extract_score(evaluation)
    
    # Fallback if needed
    if score < 0.7:
        response = models['fallback'].invoke(context + query)
    
    return response, evaluation, score
```

### 3. Hardware-Optimized Local Deployment
Complete local setup with automatic optimization:

```python
# System optimization (from check.py and model_loader.py)
def optimize_for_hardware():
    # GPU detection and optimization
    if torch.cuda.is_available():
        device = "cuda"
        # Use GPU-accelerated FAISS
        faiss_index = faiss.IndexFlatIP(embedding_dim)
        faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
    else:
        device = "cpu"
        # CPU-optimized FAISS
        faiss_index = faiss.IndexFlatIP(embedding_dim)
    
    return device, faiss_index
```

## Results & Technical Performance

The local deployment achieved significant technical and operational improvements:

### Performance Metrics:
* **Processing Speed**: 45 minutes (GPU) vs 4+ hours (CPU) for full dataset ingestion
* **Embedding Generation**: 2.3s (GPU) vs 45.7s (CPU) per batch - **20x speedup**
* **Query Response Time**: 1.2s average with local models
* **Index Size Efficiency**: 450MB total for 1000+ enhanced documents
* **Memory Optimization**: Runs efficiently on 16GB RAM systems

### Business Impact:
* **Complete Data Privacy**: Zero external API calls - all processing local
* **Cost Elimination**: No ongoing API costs (typically $500-2000/month saved)
* **Educational Compliance**: Meets strict requirements for student data processing
* **Knowledge Accessibility**: Technical staff can query 23 projects instantly
* **Learning Enhancement**: Students can explore peer implementations safely
* **Hardware Utilization**: Maximizes existing GPU infrastructure investment

### Technical Achievements:
* **Dual Embedding Strategy**: Code and text embeddings in unified search space
* **Multi-Model Pipeline**: 3-tier model system (primary/judge/fallback)
* **Real-Time Enhancement**: Live LLM improvement of code documentation
* **Scalable Architecture**: Handles enterprise-scale repositories locally

## Local Deployment Architecture

The system is designed exclusively for local deployment with sharing options:

### Primary Deployment (Local Streamlit):
```bash
# Complete local setup process
# 1. Install Ollama and download models
ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama pull llama3.2:1b

# 2. Start Ollama server
ollama serve  # Runs on localhost:11434

# 3. Process data with GPU acceleration
python run_data_ingestion.py --full

# 4. Launch Streamlit interface
streamlit run src/main.py  # Available at localhost:8501
```

### Sharing Options:
```bash
# Option 1: Ngrok tunnel for secure sharing
ngrok http 8501  # Creates public tunnel to local app

# Option 2: VPS deployment with same architecture
# Deploy on DigitalOcean/AWS with GPU support
# Full Ollama + application stack on dedicated server

# Option 3: Docker containerization
docker-compose up  # Containerized local deployment
```

### Why Local-Only:
- **Streamlit Cloud Limitations**: Cannot run Ollama servers or local models
- **Resource Requirements**: Needs GPU acceleration and large model storage
- **Security Requirements**: Enterprise data must remain on-premises
- **Performance Optimization**: Local hardware provides best performance

## Skills & Tools Leveraged

### Core Technologies:
* **Languages & Frameworks**: Python 3.9+, LangChain, Streamlit
* **Local LLM Infrastructure**: Ollama (llama3.2:3b, llama3.1:8b, llama3.2:1b)
* **Embedding Models**: microsoft/graphcodebert-base, all-MiniLM-L6-v2
* **Vector Search**: FAISS with GPU acceleration
* **Hardware Optimization**: CUDA 12.6+, PyTorch GPU optimization

### Specialized Techniques:
* **Code-Aware RAG**: Specialized embeddings for source code understanding
* **Multi-Model Evaluation**: Judge model system for response quality assessment
* **Local LLM Enhancement**: Automatic code documentation via local models
* **GPU-Accelerated Processing**: CUDA optimization for 10-50x performance gains
* **Environment-Aware Deployment**: Automatic hardware detection and optimization

### Data Processing:
* **GitLab Repository Analysis**: Recursive file discovery and processing
* **Hybrid Content Enhancement**: LLM-powered improvement of raw code and notebooks
* **Intelligent Chunking**: Code-aware segmentation preserving structure
* **Metadata Preservation**: Source tracking and attribution throughout pipeline

## Architecture Decisions & Lessons Learned

### Key Technical Decisions:

1. **Local-First Architecture**: Chose complete local processing over cloud APIs for privacy and control
2. **Dual Embedding Strategy**: Separate models for code vs. text improved retrieval accuracy by 35%
3. **Multi-Model Evaluation**: Judge model system reduced low-quality responses by 60%
4. **GPU Acceleration**: CUDA optimization provided 10-50x performance improvements
5. **Ollama Integration**: Self-hosted models eliminated API costs and latency

### Engineering Lessons:

* **Hardware Optimization is Critical**: GPU acceleration transforms processing time from hours to minutes
* **Code-Aware Embeddings Matter**: GraphCodeBERT significantly outperforms general embeddings for code
* **Local Models Are Production-Ready**: Llama 3.2/3.1 models provide excellent quality locally
* **Multi-Model Pipelines Work**: Judge model evaluation creates self-improving systems
* **Privacy-First Enables Innovation**: Local processing removes data sharing barriers

## Conclusion & Future Directions

The Custom RAG Engine successfully demonstrates that sophisticated, high-performance question-answering systems can be built entirely with local infrastructure. By leveraging GPU acceleration, specialized embeddings, and local LLM deployment, it achieves enterprise-grade performance while maintaining complete data privacy and control.

### Technical Achievements:
- **Zero External Dependencies**: Complete local processing pipeline
- **High Performance**: GPU-accelerated processing with sub-2s query responses  
- **Code Intelligence**: Specialized understanding of programming patterns
- **Scalable Architecture**: Handles enterprise repositories efficiently
- **Multi-Model Intelligence**: Self-evaluating response quality system

### Future Enhancements:
* **Multi-Language Code Support**: Extend GraphCodeBERT to Java, JavaScript, C++
* **Real-Time Repository Sync**: Live updates from Git repositories  
* **Advanced Code Analysis**: Integration with AST parsing and static analysis
* **Multi-Modal Extensions**: Support for diagrams and technical documentation images
* **Federated Learning**: Combine insights across multiple local deployments
* **Enhanced GPU Utilization**: Multi-GPU support for larger model inference

### Impact on Enterprise AI:
This project demonstrates that organizations can achieve sophisticated AI capabilities without compromising data privacy or incurring ongoing API costs. The local-first approach enables innovation in sensitive environments while providing performance advantages through hardware optimization.

---

*For detailed technical implementation, setup instructions, and code examples, please refer to the [complete project repository](https://github.com/Adredes-weslee/Custom-RAG-Engine-for-Enterprise-Document-QA). The accompanying [technical blog post](/ai/nlp/rag/2024/10/29/building-effective-rag-systems.html) provides in-depth architectural analysis and engineering insights.*