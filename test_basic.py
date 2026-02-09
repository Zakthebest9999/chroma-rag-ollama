#!/usr/bin/env python3
"""
Test script to verify the RAG system can be imported and initialized.
This is a basic sanity check that doesn't require Ollama to be running.
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        from rag_system import LocalRAGSystem
        print("✓ Successfully imported LocalRAGSystem")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_class_initialization():
    """Test that the RAG system can be initialized (without Ollama running)."""
    print("\nTesting class initialization...")
    try:
        from rag_system import LocalRAGSystem
        
        # This will initialize ChromaDB but won't call Ollama yet
        print("Creating RAGSystem instance...")
        rag = LocalRAGSystem(
            collection_name="test_collection",
            persist_directory="./test_chroma_db",
            model_name="llama2"
        )
        print(f"✓ RAG system initialized successfully")
        print(f"✓ Collection name: {rag.collection_name}")
        print(f"✓ Model name: {rag.model_name}")
        print(f"✓ Persist directory: {rag.persist_directory}")
        
        # Clean up
        import shutil
        if os.path.exists("./test_chroma_db"):
            shutil.rmtree("./test_chroma_db")
            print("✓ Cleaned up test database")
        
        return True
    except Exception as e:
        print(f"✗ Initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_method_existence():
    """Test that all expected methods exist."""
    print("\nTesting method existence...")
    try:
        from rag_system import LocalRAGSystem
        
        methods = ['add_documents', 'query', 'get_collection_count', 'delete_collection']
        all_exist = True
        
        for method in methods:
            if hasattr(LocalRAGSystem, method):
                print(f"✓ Method '{method}' exists")
            else:
                print(f"✗ Method '{method}' missing")
                all_exist = False
        
        return all_exist
    except Exception as e:
        print(f"✗ Error checking methods: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("Local RAG System - Basic Tests")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Method Existence", test_method_existence()))
    results.append(("Initialization", test_class_initialization()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
