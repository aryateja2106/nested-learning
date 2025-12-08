"""
Nested Learning Implementation
==============================

A from-scratch implementation of Google Research's "Nested Learning:
The Illusion of Deep Learning Architecture" paper.

Main Components:
- core.optimizers: DGD, M3, Delta Momentum optimizers
- core.memory: Continuum Memory System (CMS)
- models.titans: Self-Modifying Titans
- models.hope: Hope architecture (Titans + CMS)
"""

from .core.memory import CMSConfig, ContinuumMemorySystem, create_cms
from .core.optimizers import (
    DeepMomentum,
    DeltaGradientDescent,
    DeltaMomentum,
    M3Optimizer,
    create_optimizer,
)
from .models.hope import Hope, HopeConfig, HopeLayer, create_hope
from .models.titans import SelfModifyingTitans, TitansConfig, create_titans

__version__ = "0.1.0"
__author__ = "LeCoder Project"

__all__ = [
    # Optimizers
    "DeltaGradientDescent",
    "DeltaMomentum",
    "M3Optimizer",
    "DeepMomentum",
    "create_optimizer",
    # Memory
    "ContinuumMemorySystem",
    "CMSConfig",
    "create_cms",
    # Models
    "SelfModifyingTitans",
    "TitansConfig",
    "create_titans",
    "Hope",
    "HopeConfig",
    "HopeLayer",
    "create_hope",
]
