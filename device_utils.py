"""
Device utilities for cross-platform GPU support.

Supports:
- NVIDIA CUDA
- AMD ROCm (via HIP/CUDA API)
- Apple MPS
- CPU fallback
"""

import torch


def get_device(prefer_gpu: bool = True, verbose: bool = True) -> torch.device:
    """
    Get the best available compute device.
    
    Priority order:
    1. CUDA/ROCm (if available)
    2. MPS (Apple Silicon)
    3. CPU
    
    Args:
        prefer_gpu: If False, always return CPU
        verbose: Print device info
        
    Returns:
        torch.device for computation
    """
    if not prefer_gpu:
        if verbose:
            print("Using device: cpu (GPU disabled)")
        return torch.device("cpu")
    
    # Check CUDA/ROCm (ROCm uses CUDA API)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            device_name = torch.cuda.get_device_name(0)
            print(f"Using device: {device} ({device_name})")
        return device
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print(f"Using device: {device} (Apple Silicon)")
        return device
    
    # Fallback to CPU
    if verbose:
        print("Using device: cpu (no GPU available)")
    return torch.device("cpu")


def get_device_info() -> dict:
    """
    Get detailed information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        "cuda_version": None,
        "cudnn_version": None,
        "device_count": 0,
        "devices": [],
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        info["device_count"] = torch.cuda.device_count()
        
        for i in range(torch.cuda.device_count()):
            device_info = {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": torch.cuda.get_device_properties(i).total_memory / (1024**3),
                "compute_capability": f"{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}",
            }
            info["devices"].append(device_info)
    
    return info


def print_device_info():
    """Print formatted device information."""
    info = get_device_info()
    
    print("=" * 50)
    print("PyTorch Device Information")
    print("=" * 50)
    print(f"PyTorch version: {info['pytorch_version']}")
    print(f"CUDA/ROCm available: {info['cuda_available']}")
    print(f"MPS available: {info['mps_available']}")
    
    if info['cuda_available']:
        print(f"CUDA version: {info['cuda_version']}")
        print(f"cuDNN version: {info['cudnn_version']}")
        print(f"Device count: {info['device_count']}")
        
        for dev in info['devices']:
            print(f"\nDevice {dev['index']}: {dev['name']}")
            print(f"  Memory: {dev['total_memory_gb']:.1f} GB")
            print(f"  Compute capability: {dev['compute_capability']}")
    
    print("=" * 50)


if __name__ == "__main__":
    print_device_info()
    print()
    device = get_device()
    
    # Quick benchmark
    print("\nRunning quick benchmark...")
    import time
    
    sizes = [1000, 2000, 4000]
    for size in sizes:
        x = torch.randn(size, size, device=device)
        
        # Warmup
        _ = torch.matmul(x, x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Timed run
        start = time.perf_counter()
        for _ in range(10):
            _ = torch.matmul(x, x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        print(f"  {size}x{size} matmul (10x): {elapsed*1000:.1f} ms")
