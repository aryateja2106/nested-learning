"""
Experiments Package
==================

Enterprise experiments showcasing HOPE model capabilities and LeCoder cGPU CLI integration.

This package contains:
- enterprise_pipeline.py: Business use case for continual learning
- cuda_kernels.py: CUDA-accelerated operations for A100 optimization
"""

from .enterprise_pipeline import EnterprisePipeline, EnterpriseConfig
from .cuda_kernels import (
    fused_titans_update,
    optimized_newton_schulz,
    parallel_cms_update,
    check_cuda_available,
)

__all__ = [
    "EnterprisePipeline",
    "EnterpriseConfig",
    "fused_titans_update",
    "optimized_newton_schulz",
    "parallel_cms_update",
    "check_cuda_available",
]


