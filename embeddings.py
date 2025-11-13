# embeddings.py
from functools import lru_cache
from typing import List
import os

# CRITICAL: Force CPU-only mode BEFORE any PyTorch imports
# This must be set before torch is imported anywhere
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Disable MPS by setting this BEFORE importing torch
import sys
def _disable_mps():
    """Monkey-patch to force CPU usage"""
    import torch
    # Override MPS availability check
    if hasattr(torch.backends, 'mps'):
        torch.backends.mps.is_available = lambda: False
        torch.backends.mps.is_built = lambda: False
    # Set default device to CPU
    torch.set_default_device('cpu')

# We use the local HF cache; first run downloads, then fully offline.
# Model: nomic-ai/nomic-embed-text-v1.5
# Note: This model requires trust_remote_code=True and einops package
NOMIC_MODEL_ID = os.getenv("NOMIC_MODEL_ID", "nomic-ai/nomic-embed-text-v1.5")

@lru_cache(maxsize=1)
def _load_model():
    import torch
    from sentence_transformers import SentenceTransformer
    
    # Force disable MPS
    _disable_mps()
    
    # Clear any existing GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    device = "cpu"
    print(f"🔧 Loading embedding model on CPU (MPS disabled to avoid memory issues)...")
    
    # Nomic models require trust_remote_code and use custom tokenizers
    model = SentenceTransformer(
        NOMIC_MODEL_ID, 
        trust_remote_code=True,
        device=device,
        model_kwargs={"trust_remote_code": True}
    )
    
    # Force model to CPU even if it tries to move to GPU
    model = model.to('cpu')
    
    print(f"✓ Model loaded on device: {model.device}")
    return model

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed texts in batches to avoid memory overflow.
    For large document sets, processing all chunks at once can exceed memory.
    Uses CPU to avoid MPS memory issues on Mac.
    """
    if not texts:
        return []
    
    model = _load_model()
    
    # Very conservative batch size for CPU processing and memory management
    BATCH_SIZE = 16  # Reduced from 32 to use less memory
    all_embeddings = []
    
    print(f"Embedding {len(texts)} chunks in batches of {BATCH_SIZE} (CPU mode)...")
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_embeddings = model.encode(
            batch, 
            normalize_embeddings=True, 
            show_progress_bar=False,
            batch_size=BATCH_SIZE,
            device="cpu"  # Force CPU
        )
        all_embeddings.extend(batch_embeddings.tolist())
        
        # Progress indicator for large batches
        if (i + BATCH_SIZE) % 100 == 0:
            print(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)} chunks...")
    
    print(f"✓ Completed embedding {len(texts)} chunks")
    return all_embeddings

def embed_one(text: str) -> List[float]:
    return embed_texts([text])[0]
