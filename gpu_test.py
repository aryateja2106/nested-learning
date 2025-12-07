import torch
import time

def matrix_multiply_gpu():
    """Perform matrix multiplication on GPU"""
    print("🚀 Starting GPU computation...")
    
    # Create large matrices
    size = 5000
    a = torch.randn(size, size).cuda()
    b = torch.randn(size, size).cuda()
    
    # Time the computation
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()  # Wait for GPU to finish
    end = time.time()
    
    print(f"✅ Matrix multiplication ({size}x{size}) completed!")
    print(f"⏱️  Time taken: {end - start:.4f} seconds")
    print(f"📊 Result shape: {c.shape}")
    print(f"🎯 Sample result value: {c[0, 0].item():.4f}")
    
    return c

def gpu_info():
    """Display GPU information"""
    print("=" * 50)
    print("GPU INFORMATION")
    print("=" * 50)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("=" * 50)

if __name__ == "__main__":
    gpu_info()
    result = matrix_multiply_gpu()
    print("\n✨ GPU computation successful!")
