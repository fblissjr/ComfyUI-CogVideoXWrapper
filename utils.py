import importlib.metadata
import torch
import logging
import importlib.metadata
from functools import lru_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def check_diffusers_version():
    try:
        version = importlib.metadata.version('diffusers')
        required_version = '0.31.0'
        if version < required_version:
            raise AssertionError(f"diffusers version {version} is installed, but version {required_version} or higher is required.")
    except importlib.metadata.PackageNotFoundError:
        raise AssertionError("diffusers is not installed.")
    
def remove_specific_blocks(model, block_indices_to_remove):
    import torch.nn as nn
    transformer_blocks = model.transformer_blocks
    new_blocks = [block for i, block in enumerate(transformer_blocks) if i not in block_indices_to_remove]
    model.transformer_blocks = nn.ModuleList(new_blocks)
    
    return model

def optimize_memory_usage():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
def get_optimal_device_settings():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return device, dtype
    
def setup_torch_compile(model):
    if not torch.cuda.is_available():
        return model
        
    try:
        compile_config = {
            "backend": "inductor",
            "mode": "max-autotune",
            "fullgraph": True,
            "dynamic": False,
            "options": {
                "epilogue_fusion": True,
                "max_autotune": True
            }
        }
        return torch.compile(model, **compile_config)
    except Exception as e:
        print(f"Torch compile failed: {e}")
        return model
    
# Add new optimization utilities
@lru_cache(maxsize=8)
def cached_fft(tensor_key, tensor):
    """Cached FFT computation"""
    tensor_fft = torch.fft.fft2(tensor)
    return torch.fft.fftshift(tensor_fft)

def optimize_memory():
    """Memory optimization utility"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_optimal_dtype():
    """Get optimal dtype based on device"""
    if torch.cuda.is_available():
        if torch.cuda.get_device_capability()[0] >= 7:
            return torch.float16
    return torch.float32