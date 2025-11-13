#!/usr/bin/env python3
"""
ChronoMed Dependency Checker
Verifies all required packages are installed correctly
"""

import sys
import subprocess

print("=" * 80)
print("ChronoMed Dependency Checker")
print("=" * 80)
print()

# Get Python info
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print()

# List of required packages
required_packages = [
    ("streamlit", "Streamlit UI framework"),
    ("pandas", "Data manipulation"),
    ("requests", "HTTP client"),
    ("chromadb", "Vector database"),
    ("sentence_transformers", "Local embeddings"),
    ("langchain_community", "LangChain Ollama integration"),
    ("docling", "PDF parsing library"),
    ("torch", "PyTorch (required by Docling)"),
    ("accelerate", "PyTorch acceleration (required by Docling)"),
]

print("Checking required packages:")
print("-" * 80)

all_ok = True
for package, description in required_packages:
    try:
        if package == "langchain_community":
            mod = __import__("langchain_community")
        else:
            mod = __import__(package)
        
        version = getattr(mod, "__version__", "unknown")
        print(f"✓ {package:25} {version:15} - {description}")
    except ImportError as e:
        print(f"✗ {package:25} {'NOT INSTALLED':15} - {description}")
        print(f"  Error: {e}")
        all_ok = False

print()
print("=" * 80)

if all_ok:
    print("✅ All dependencies installed correctly!")
    print()
    print("Testing Docling import...")
    try:
        from docling.document_converter import DocumentConverter
        print("✓ Docling DocumentConverter imported successfully")
        print()
        print("System is ready! Run Streamlit with:")
        print(f"  {sys.executable} -m streamlit run ChronoMed_AI.py")
    except Exception as e:
        print(f"✗ Docling import failed: {e}")
        print()
        print("This might be a runtime issue. Try setting CPU-only mode:")
        print("  export CUDA_VISIBLE_DEVICES=''")
else:
    print("❌ Missing dependencies detected!")
    print()
    print("Install missing packages with:")
    print(f"  {sys.executable} -m pip install <package_name>")

print("=" * 80)
