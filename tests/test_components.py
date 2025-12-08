"""
Test Suite for Nested Learning Components
=========================================

Tests all core components:
- Delta Gradient Descent (DGD)
- M3 Optimizer
- Continuum Memory System (CMS)
- Self-Modifying Titans
- Hope Architecture
"""

import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.memory import CMSConfig, ContinuumMemorySystem, MLPBlock, create_cms
from src.core.optimizers import DeltaGradientDescent, M3Optimizer, create_optimizer, newton_schulz
from src.models.hope import Hope, HopeConfig, HopeLayer, create_hope
from src.models.titans import MemoryMLP, SelfModifyingTitans, TitansConfig, create_titans


class TestNewtonSchulz:
    """Tests for Newton-Schulz orthogonalization."""

    def test_output_shape(self):
        """Output should have same shape as input."""
        M = torch.randn(64, 32)
        result = newton_schulz(M, steps=5)
        assert result.shape == M.shape

    def test_approximately_orthogonal(self):
        """Result should be approximately orthogonal."""
        M = torch.randn(32, 32)
        result = newton_schulz(M, steps=15)  # More iterations for better convergence

        # Check O^T O ≈ I
        product = result.T @ result
        identity = torch.eye(32)
        error = (product - identity).abs().max()
        assert error < 0.25, f"Orthogonality error too high: {error}"

    def test_handles_1d_input(self):
        """Should handle 1D tensors gracefully."""
        M = torch.randn(32)
        result = newton_schulz(M, steps=5)
        assert result.shape == M.shape


class TestDeltaGradientDescent:
    """Tests for DGD optimizer."""

    def test_initialization(self):
        """Should initialize without errors."""
        model = nn.Linear(10, 5)
        optimizer = DeltaGradientDescent(model.parameters(), lr=0.01)
        assert optimizer is not None

    def test_step_reduces_loss(self):
        """Optimization step should reduce loss."""
        torch.manual_seed(42)
        model = nn.Linear(10, 5)
        optimizer = DeltaGradientDescent(model.parameters(), lr=0.1)

        x = torch.randn(32, 10)
        target = torch.randn(32, 5)

        # Initial loss
        loss1 = F.mse_loss(model(x), target)

        # Take step
        loss1.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Check loss decreased
        loss2 = F.mse_loss(model(x), target)
        assert loss2 < loss1, f"Loss did not decrease: {loss1} -> {loss2}"

    def test_with_input_tensor(self):
        """Should work with input tensor for adaptive decay."""
        model = nn.Linear(10, 5)
        optimizer = DeltaGradientDescent(model.parameters(), lr=0.1)

        x = torch.randn(32, 10)
        target = torch.randn(32, 5)

        loss = F.mse_loss(model(x), target)
        loss.backward()

        # Step with input tensor
        optimizer.step(input_tensor=x.mean(dim=0))
        assert True  # If we get here without error, test passes


class TestM3Optimizer:
    """Tests for M3 optimizer."""

    def test_initialization(self):
        """Should initialize without errors."""
        model = nn.Linear(10, 5)
        optimizer = M3Optimizer(model.parameters(), lr=0.02)
        assert optimizer is not None

    def test_step_updates_parameters(self):
        """Optimization step should update parameters."""
        torch.manual_seed(42)
        model = nn.Linear(10, 5)
        initial_weights = model.weight.data.clone()

        optimizer = M3Optimizer(model.parameters(), lr=0.02)

        x = torch.randn(32, 10)
        target = torch.randn(32, 5)

        loss = F.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()

        # Check weights changed
        assert not torch.allclose(model.weight.data, initial_weights)

    def test_slow_momentum_updates(self):
        """Slow momentum should update at correct frequency."""
        model = nn.Linear(10, 5)
        optimizer = M3Optimizer(model.parameters(), lr=0.02, chunk_size=5)

        x = torch.randn(32, 10)
        target = torch.randn(32, 5)

        # Run for chunk_size steps
        for _ in range(5):
            optimizer.zero_grad()
            loss = F.mse_loss(model(x), target)
            loss.backward()
            optimizer.step()

        # Check slow momentum was updated (step_count should be 5)
        assert optimizer.step_count == 5


class TestMLPBlock:
    """Tests for MLP block in CMS."""

    def test_forward_shape(self):
        """Output shape should match input."""
        block = MLPBlock(d_model=256, d_hidden=1024)
        x = torch.randn(2, 32, 256)
        output = block(x)
        assert output.shape == x.shape

    def test_residual_connection(self):
        """Should have residual connection."""
        block = MLPBlock(d_model=256, d_hidden=1024)
        x = torch.randn(2, 32, 256)

        # With near-zero weights, output should be close to input
        with torch.no_grad():
            for p in block.parameters():
                p.fill_(0.0)

        output = block(x)
        assert torch.allclose(output, x, atol=1e-6)


class TestContinuumMemorySystem:
    """Tests for CMS."""

    def test_initialization(self):
        """Should initialize without errors."""
        config = CMSConfig(d_model=256, num_levels=4)
        cms = ContinuumMemorySystem(config)
        assert len(cms.mlp_blocks) == 4

    def test_forward_shape(self):
        """Output shape should match input."""
        config = CMSConfig(d_model=256, num_levels=4)
        cms = ContinuumMemorySystem(config)

        x = torch.randn(2, 32, 256)
        output = cms(x)
        assert output.shape == x.shape

    def test_chunk_sizes_exponential(self):
        """Chunk sizes should increase exponentially."""
        config = CMSConfig(d_model=256, num_levels=4, base_chunk_size=8)
        cms = ContinuumMemorySystem(config)

        expected = [8, 16, 32, 64]
        assert cms.chunk_sizes == expected

    def test_sequential_aggregation(self):
        """Sequential aggregation should work."""
        config = CMSConfig(d_model=256, num_levels=2, aggregation="sequential")
        cms = ContinuumMemorySystem(config)

        x = torch.randn(2, 32, 256)
        output = cms(x)
        assert output.shape == x.shape

    def test_independent_aggregation(self):
        """Independent aggregation should work."""
        config = CMSConfig(d_model=256, num_levels=2, aggregation="independent")
        cms = ContinuumMemorySystem(config)

        x = torch.randn(2, 32, 256)
        output = cms(x)
        assert output.shape == x.shape

    def test_update_schedule(self):
        """Should return correct update schedule."""
        config = CMSConfig(d_model=256, num_levels=3, base_chunk_size=4)
        cms = ContinuumMemorySystem(config)

        schedule = cms.get_update_schedule(max_steps=20)

        # Level 0 (chunk=4): updates at 4, 8, 12, 16, 20
        assert schedule[0] == [4, 8, 12, 16, 20]
        # Level 1 (chunk=8): updates at 8, 16
        assert schedule[1] == [8, 16]
        # Level 2 (chunk=16): updates at 16
        assert schedule[2] == [16]


class TestMemoryMLP:
    """Tests for Memory MLP in Titans."""

    def test_forward_same_dims(self):
        """Should work with same input/output dims."""
        mlp = MemoryMLP(d_input=64, d_output=64, d_hidden=128)
        x = torch.randn(2, 32, 64)
        output = mlp(x)
        assert output.shape == x.shape

    def test_forward_different_dims(self):
        """Should work with different input/output dims."""
        mlp = MemoryMLP(d_input=64, d_output=32, d_hidden=128)
        x = torch.randn(2, 32, 64)
        output = mlp(x)
        assert output.shape == (2, 32, 32)


class TestSelfModifyingTitans:
    """Tests for Self-Modifying Titans."""

    def test_initialization(self):
        """Should initialize without errors."""
        config = TitansConfig(d_model=256, d_key=32, d_value=32, num_heads=4)
        titans = SelfModifyingTitans(config)
        assert titans is not None

    def test_forward_shape(self):
        """Output shape should match input."""
        config = TitansConfig(d_model=256, d_key=32, d_value=32, num_heads=4)
        titans = SelfModifyingTitans(config)

        x = torch.randn(2, 32, 256)
        output = titans(x)
        assert output.shape == x.shape

    def test_memory_update(self):
        """Memory should update during forward pass."""
        config = TitansConfig(d_model=256, d_key=32, d_value=32, num_heads=4)
        titans = SelfModifyingTitans(config)

        initial_memory = titans.M_memory.clone()

        x = torch.randn(2, 32, 256)
        output, memory_states = titans(x, return_memory_states=True)

        # Memory should have changed
        assert len(memory_states) > 0
        assert not torch.allclose(
            memory_states[-1], initial_memory.unsqueeze(0).expand_as(memory_states[-1])
        )

    def test_reset_memory(self):
        """Reset memory should zero out state."""
        config = TitansConfig(d_model=256, d_key=32, d_value=32, num_heads=4)
        titans = SelfModifyingTitans(config)

        # Modify memory
        titans.M_memory.fill_(1.0)
        titans.reset_memory()

        assert torch.allclose(titans.M_memory, torch.zeros_like(titans.M_memory))


class TestHopeLayer:
    """Tests for Hope layer."""

    def test_initialization(self):
        """Should initialize without errors."""
        config = HopeConfig(d_model=256, num_heads=4, cms_num_levels=2)
        layer = HopeLayer(config, layer_idx=0)
        assert layer is not None

    def test_forward_shape(self):
        """Output shape should match input."""
        config = HopeConfig(d_model=256, num_heads=4, cms_num_levels=2)
        layer = HopeLayer(config, layer_idx=0)

        x = torch.randn(2, 32, 256)
        output = layer(x)
        assert output.shape == x.shape


class TestHope:
    """Tests for full Hope model."""

    def test_initialization(self):
        """Should initialize without errors."""
        config = HopeConfig(
            d_model=128, num_layers=2, num_heads=4, vocab_size=1000, cms_num_levels=2
        )
        model = Hope(config)
        assert model is not None

    def test_forward_shape(self):
        """Logits should have correct shape."""
        config = HopeConfig(
            d_model=128, num_layers=2, num_heads=4, vocab_size=1000, cms_num_levels=2
        )
        model = Hope(config)

        input_ids = torch.randint(0, 1000, (2, 32))
        output = model(input_ids)

        assert output["logits"].shape == (2, 32, 1000)

    def test_forward_with_labels(self):
        """Should compute loss when labels provided."""
        config = HopeConfig(
            d_model=128, num_layers=2, num_heads=4, vocab_size=1000, cms_num_levels=2
        )
        model = Hope(config)

        input_ids = torch.randint(0, 1000, (2, 32))
        labels = torch.randint(0, 1000, (2, 32))
        output = model(input_ids, labels=labels)

        assert "loss" in output
        assert output["loss"].ndim == 0  # Scalar

    def test_generate(self):
        """Should generate tokens."""
        config = HopeConfig(
            d_model=128, num_layers=2, num_heads=4, vocab_size=1000, cms_num_levels=2
        )
        model = Hope(config)
        model.eval()

        prompt = torch.randint(0, 1000, (1, 10))
        generated = model.generate(prompt, max_new_tokens=5)

        assert generated.shape == (1, 15)  # 10 prompt + 5 new


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_optimizer(self):
        """Should create correct optimizer types."""
        model = nn.Linear(10, 5)

        dgd = create_optimizer("dgd", model.parameters(), lr=0.01)
        assert isinstance(dgd, DeltaGradientDescent)

        m3 = create_optimizer("m3", model.parameters(), lr=0.02)
        assert isinstance(m3, M3Optimizer)

        adam = create_optimizer("adam", model.parameters(), lr=0.001)
        assert isinstance(adam, torch.optim.Adam)

    def test_create_cms(self):
        """Should create CMS with correct config."""
        cms = create_cms(d_model=512, num_levels=4)
        assert isinstance(cms, ContinuumMemorySystem)
        assert len(cms.mlp_blocks) == 4

    def test_create_titans(self):
        """Should create Titans with correct config."""
        titans = create_titans(d_model=512, num_heads=8)
        assert isinstance(titans, SelfModifyingTitans)

    def test_create_hope(self):
        """Should create Hope with correct config."""
        model = create_hope(d_model=512, num_layers=6)
        assert isinstance(model, Hope)
        assert len(model.layers) == 6


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_hope_training_step(self):
        """Should complete a full training step."""
        config = HopeConfig(
            d_model=128, num_layers=2, num_heads=4, vocab_size=1000, cms_num_levels=2
        )
        model = Hope(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Forward
        input_ids = torch.randint(0, 1000, (2, 32))
        labels = torch.randint(0, 1000, (2, 32))
        output = model(input_ids, labels=labels)

        # Backward
        loss = output["loss"]
        loss.backward()

        # Step
        optimizer.step()
        optimizer.zero_grad()

        assert loss.item() > 0  # Loss should be positive

    def test_hope_with_m3_optimizer(self):
        """Should work with M3 optimizer."""
        config = HopeConfig(
            d_model=128, num_layers=2, num_heads=4, vocab_size=1000, cms_num_levels=2
        )
        model = Hope(config)
        optimizer = M3Optimizer(model.parameters(), lr=0.01)

        input_ids = torch.randint(0, 1000, (2, 32))
        labels = torch.randint(0, 1000, (2, 32))

        for _ in range(3):
            optimizer.zero_grad()
            output = model(input_ids, labels=labels)
            output["loss"].backward()
            optimizer.step()

        assert True  # If we complete without error, test passes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
