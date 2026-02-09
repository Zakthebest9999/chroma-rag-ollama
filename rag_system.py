#!/usr/bin/env python3
"""
Local RAG System using Chroma and Ollama

This module implements a Retrieval Augmented Generation (RAG) system
that runs completely locally using:
- ChromaDB for vector storage
- Ollama for local LLM inference
"""

import os
from typing import List, Optional
import chromadb
from chromadb.config import Settings
import ollama


class LocalRAGSystem:
    """A local RAG system using ChromaDB and Ollama."""
    
    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./chroma_db",
        model_name: str = "llama2"
    ):
        """
        Initialize the RAG system.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector database
            model_name: Ollama model to use for generation
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.model_name = model_name
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(name=collection_name)
            print(f"Created new collection: {collection_name}")
    
    def add_documents(self, documents: List[str], ids: Optional[List[str]] = None):
        """
        Add documents to the vector database.
        
        Args:
            documents: List of document texts to add
            ids: Optional list of document IDs (auto-generated if not provided)
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        self.collection.add(
            documents=documents,
            ids=ids
        )
        print(f"Added {len(documents)} documents to the collection")
    
    def query(self, question: str, n_results: int = 3) -> dict:
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to ask
            n_results: Number of relevant documents to retrieve
            
        Returns:
            Dictionary containing the answer and retrieved documents
        """
        # Retrieve relevant documents
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )
        
        # Extract retrieved documents
        retrieved_docs = results['documents'][0] if results['documents'] else []
        
        # Build context from retrieved documents
        context = "\n\n".join([f"Document {i+1}:\n{doc}" 
                               for i, doc in enumerate(retrieved_docs)])
        
        # Create prompt for the LLM
        prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        # Generate answer using Ollama
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt
            )
            answer = response['response']
        except Exception as e:
            answer = f"Error generating response: {str(e)}"
        
        return {
            'question': question,
            'answer': answer,
            'retrieved_documents': retrieved_docs,
            'n_results': len(retrieved_docs)
        }
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()
    
    def delete_collection(self):
        """Delete the current collection."""
        self.client.delete_collection(name=self.collection_name)
        print(f"Deleted collection: {self.collection_name}")


def main():
    """Example usage of the RAG system."""
    # Initialize the RAG system
    rag = LocalRAGSystem(
        collection_name="my_documents",
        persist_directory="./chroma_db",
        model_name="llama2"
    )
    
    # Check if collection is empty
    if rag.get_collection_count() == 0:
        print("\nAdding sample documents...")
        # Add sample documents
        sample_docs = [
            "Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
            "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            "RAG (Retrieval Augmented Generation) is a technique that combines information retrieval with text generation to provide more accurate and contextual responses.",
            "ChromaDB is an open-source embedding database that makes it easy to build LLM applications with embeddings.",
            "Ollama is a tool that allows you to run large language models locally on your own machine."
        ]
        rag.add_documents(sample_docs)
    else:
        print(f"\nCollection already contains {rag.get_collection_count()} documents")
    
    # Example query
    print("\n" + "="*50)
    print("Example Query")
    print("="*50)
    question = "What is RAG?"
    result = rag.query(question)
    
    print(f"\nQuestion: {result['question']}")
    print(f"\nRetrieved {result['n_results']} relevant documents:")
    for i, doc in enumerate(result['retrieved_documents'], 1):
        print(f"\n{i}. {doc[:100]}...")
    print(f"\nAnswer:\n{result['answer']}")


if __name__ == "__main__":
    main()
