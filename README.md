# Chroma RAG Ollama - Local RAG Solution

A complete local Retrieval Augmented Generation (RAG) system using ChromaDB for vector storage and Ollama for local LLM inference.

## Features

- 🚀 **Fully Local**: Runs entirely on your machine, no API keys needed
- 📚 **Vector Database**: Uses ChromaDB for efficient document storage and retrieval
- 🤖 **Local LLM**: Powered by Ollama for response generation
- 💾 **Persistent Storage**: Automatic persistence of your document collection
- 🔍 **Semantic Search**: Finds relevant documents using embeddings
- 📝 **Easy to Use**: Simple Python API for adding documents and asking questions

## Prerequisites

Before you begin, ensure you have the following installed:

1. **Python 3.8+**: [Download Python](https://www.python.org/downloads/)
2. **Ollama**: [Install Ollama](https://ollama.ai/)

After installing Ollama, pull a model (e.g., llama2):
```bash
ollama pull llama2
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Zakthebest9999/chroma-rag-ollama.git
cd chroma-rag-ollama
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

Run the main RAG system with sample documents:
```bash
python rag_system.py
```

This will:
- Initialize the RAG system
- Add sample documents about Python, ML, RAG, ChromaDB, and Ollama
- Ask an example question and display the answer

### Custom Documents Example

Run the example script with custom documents:
```bash
python example.py
```

### Interactive Mode

Ask questions interactively:
```bash
python example.py interactive
```

## Usage Guide

### 1. Initialize the RAG System

```python
from rag_system import LocalRAGSystem

# Create a RAG instance
rag = LocalRAGSystem(
    collection_name="my_documents",
    persist_directory="./chroma_db",
    model_name="llama2"  # or any other Ollama model
)
```

### 2. Add Documents

```python
# Add documents to the knowledge base
documents = [
    "Your first document text here...",
    "Your second document text here...",
    "More documents..."
]

rag.add_documents(documents)
```

### 3. Query the System

```python
# Ask a question
result = rag.query("What is your question?", n_results=3)

print(f"Answer: {result['answer']}")
print(f"Retrieved documents: {result['retrieved_documents']}")
```

## How It Works

1. **Document Ingestion**: Documents are added to ChromaDB, which automatically creates embeddings
2. **Query Processing**: When you ask a question:
   - ChromaDB finds the most relevant documents using semantic search
   - The retrieved documents are used as context
   - Ollama generates an answer based on the context
3. **Response**: You get an answer grounded in your documents

## Configuration

### Change the LLM Model

You can use any model available in Ollama:
```python
rag = LocalRAGSystem(model_name="mistral")  # or "codellama", "neural-chat", etc.
```

### Adjust Number of Retrieved Documents

```python
result = rag.query("Your question?", n_results=5)  # Retrieve top 5 documents
```

## Project Structure

```
chroma-rag-ollama/
├── rag_system.py      # Main RAG system implementation
├── example.py         # Example usage scripts
├── requirements.txt   # Python dependencies
├── README.md          # This file
└── chroma_db/         # ChromaDB storage (auto-created)
```

## Dependencies

- `chromadb`: Vector database for storing and retrieving documents
- `langchain` & `langchain-community`: Framework for LLM applications
- `ollama`: Python client for Ollama LLM

## Troubleshooting

### Ollama Connection Error
- Make sure Ollama is running: `ollama serve`
- Verify the model is pulled: `ollama list`

### ChromaDB Issues
- Delete the `chroma_db` directory to start fresh
- Make sure you have write permissions in the project directory

## Advanced Usage

### Custom Collection Management

```python
# Get collection count
count = rag.get_collection_count()
print(f"Documents in collection: {count}")

# Delete a collection (use with caution!)
rag.delete_collection()
```

### Using Different Collections

```python
# Create separate collections for different topics
tech_rag = LocalRAGSystem(collection_name="tech_docs")
medical_rag = LocalRAGSystem(collection_name="medical_docs")
```

## License

MIT License - feel free to use this project for your own purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [ChromaDB](https://www.trychroma.com/) - The AI-native open-source embedding database
- [Ollama](https://ollama.ai/) - Get up and running with large language models locally
- [LangChain](https://www.langchain.com/) - Framework for developing LLM applications
