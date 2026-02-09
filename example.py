#!/usr/bin/env python3
"""
Example script demonstrating how to use the Local RAG System.
"""

from rag_system import LocalRAGSystem


def example_custom_documents():
    """Example of using the RAG system with custom documents."""
    print("="*60)
    print("Local RAG System - Custom Documents Example")
    print("="*60)
    
    # Initialize RAG system
    rag = LocalRAGSystem(
        collection_name="technical_docs",
        persist_directory="./chroma_db",
        model_name="llama2"  # Change to your preferred Ollama model
    )
    
    # Add custom documents about a specific topic
    documents = [
        "Docker is a platform that enables developers to package applications "
        "into containers—standardized executable components combining application "
        "source code with the operating system libraries and dependencies required "
        "to run that code in any environment.",
        
        "Kubernetes is an open-source container orchestration platform that "
        "automates the deployment, scaling, and management of containerized applications. "
        "It was originally designed by Google and is now maintained by the Cloud Native "
        "Computing Foundation.",
        
        "Microservices architecture is an approach to developing a single application "
        "as a suite of small services, each running in its own process and communicating "
        "with lightweight mechanisms, often an HTTP resource API.",
        
        "CI/CD stands for Continuous Integration and Continuous Deployment. It is "
        "a method to frequently deliver apps to customers by introducing automation "
        "into the stages of app development.",
    ]
    
    print(f"\nAdding {len(documents)} documents to the collection...")
    rag.add_documents(documents)
    print(f"Total documents in collection: {rag.get_collection_count()}")
    
    # Ask questions
    questions = [
        "What is Docker?",
        "How does Kubernetes help with containers?",
        "What is the purpose of CI/CD?",
    ]
    
    for question in questions:
        print("\n" + "="*60)
        print(f"Question: {question}")
        print("="*60)
        
        result = rag.query(question, n_results=2)
        
        print(f"\nRetrieved {result['n_results']} relevant documents:")
        for i, doc in enumerate(result['retrieved_documents'], 1):
            print(f"\n{i}. {doc[:150]}...")
        
        print(f"\nAnswer:\n{result['answer']}")


def example_interactive_mode():
    """Interactive mode for asking questions."""
    print("="*60)
    print("Local RAG System - Interactive Mode")
    print("="*60)
    
    rag = LocalRAGSystem(
        collection_name="my_documents",
        persist_directory="./chroma_db",
        model_name="llama2"
    )
    
    print(f"\nCurrent collection has {rag.get_collection_count()} documents")
    print("\nType 'quit' or 'exit' to stop")
    print("Type 'add' to add a new document")
    print("="*60)
    
    while True:
        user_input = input("\nYour question: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'add':
            doc = input("Enter document text: ").strip()
            if doc:
                rag.add_documents([doc])
                print("Document added successfully!")
            continue
        
        if not user_input:
            continue
        
        result = rag.query(user_input)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\n(Based on {result['n_results']} retrieved documents)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        example_interactive_mode()
    else:
        example_custom_documents()
