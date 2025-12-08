"""
Nested Learning Interactive Demo
================================

A Gradio-based demo for exploring the Nested Learning components:
- Delta Gradient Descent (DGD) visualization
- Continuum Memory System (CMS) behavior
- Hope model inference
"""

# Import components
import sys
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

# Import components to validate module availability (used for validation only)
try:
    from src.core.memory import CMSConfig, ContinuumMemorySystem  # noqa: F401
    from src.core.optimizers import DeltaGradientDescent, M3Optimizer  # noqa: F401
    from src.models.hope import Hope, HopeConfig  # noqa: F401
    from src.models.titans import SelfModifyingTitans, TitansConfig  # noqa: F401

    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Nested Learning modules: {e}")
    MODULES_AVAILABLE = False


def visualize_cms_update_schedule(
    num_levels: int = 4, base_chunk_size: int = 8, total_steps: int = 64
) -> plt.Figure:
    """Visualize when each CMS level updates."""
    fig, ax = plt.subplots(figsize=(12, 4))

    colors = plt.cm.viridis(np.linspace(0, 1, num_levels))

    for level in range(num_levels):
        chunk_size = base_chunk_size * (2**level)
        update_steps = list(range(chunk_size, total_steps + 1, chunk_size))

        for step in update_steps:
            ax.axvline(x=step, color=colors[level], alpha=0.7, linewidth=2)

        ax.plot(
            [], [], color=colors[level], linewidth=2, label=f"Level {level} (chunk={chunk_size})"
        )

    ax.set_xlim(0, total_steps)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Update Events", fontsize=12)
    ax.set_title("CMS Update Schedule: Multi-Scale Memory Updates", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_dgd_decay(
    learning_rate: float = 0.1,
    normalized_input: bool = True,
    input_dim: int = 10,
    num_steps: int = 50,
) -> plt.Figure:
    """Visualize DGD's adaptive decay behavior."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Initialize weight matrix
    W = torch.randn(input_dim, input_dim) * 0.1
    W_history = [W.clone()]
    decay_factors = []

    eta_prime = learning_rate / (1 + learning_rate) if normalized_input else learning_rate

    for _t in range(num_steps):
        # Random normalized input
        x = torch.randn(input_dim)
        x = x / x.norm()

        # Compute decay factor (I - η' x x^T)
        outer = torch.outer(x, x)
        decay = torch.eye(input_dim) - eta_prime * outer
        decay_factors.append(decay.diagonal().mean().item())

        # Update (simplified - no gradient term for visualization)
        W = W @ decay
        W_history.append(W.clone())

    # Plot 1: Weight magnitude over time
    weight_norms = [w.norm().item() for w in W_history]
    ax1.plot(weight_norms, "b-", linewidth=2)
    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Weight Matrix Norm", fontsize=12)
    ax1.set_title("DGD: Adaptive Weight Decay", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Decay factors
    ax2.plot(decay_factors, "r-", linewidth=2)
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Step", fontsize=12)
    ax2.set_ylabel("Avg Decay Factor", fontsize=12)
    ax2.set_title("Input-Dependent Decay (I - η' x x^T)", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_memory_compression(
    seq_len: int = 64, d_model: int = 32, d_memory: int = 16
) -> plt.Figure:
    """Visualize how associative memory compresses information."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Generate random sequence (simulating context)
    context = torch.randn(seq_len, d_model)

    # Simulate compression into memory
    memory = torch.zeros(d_memory, d_model)
    attention_patterns = []

    for t in range(seq_len):
        # Simplified attention-based compression
        context[t, :d_memory]
        value = context[t]
        attn = F.softmax(torch.randn(d_memory) * 0.5, dim=0)
        attention_patterns.append(attn.numpy())
        memory = 0.95 * memory + torch.outer(attn, value) * 0.05

    # Plot 1: Input sequence heatmap
    im1 = axes[0].imshow(context.numpy().T, aspect="auto", cmap="coolwarm")
    axes[0].set_xlabel("Token Position")
    axes[0].set_ylabel("Feature Dimension")
    axes[0].set_title("Input Context")
    plt.colorbar(im1, ax=axes[0])

    # Plot 2: Attention patterns over time
    attention_matrix = np.array(attention_patterns)
    im2 = axes[1].imshow(attention_matrix.T, aspect="auto", cmap="viridis")
    axes[1].set_xlabel("Token Position")
    axes[1].set_ylabel("Memory Slot")
    axes[1].set_title("Attention to Memory")
    plt.colorbar(im2, ax=axes[1])

    # Plot 3: Final memory state
    im3 = axes[2].imshow(memory.numpy(), aspect="auto", cmap="coolwarm")
    axes[2].set_xlabel("Feature Dimension")
    axes[2].set_ylabel("Memory Slot")
    axes[2].set_title("Compressed Memory State")
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    return fig


def visualize_nested_learning_paradigm() -> plt.Figure:
    """Visualize the Nested Learning paradigm structure."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Level colors
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

    # Draw nested levels
    levels = [
        {"name": "Level 0: Pre-Training\n(Lowest Frequency)", "y": 8.5, "width": 9},
        {"name": "Level 1: Slow Memory\n(Low Frequency)", "y": 6.5, "width": 7},
        {"name": "Level 2: Fast Memory\n(High Frequency)", "y": 4.5, "width": 5},
        {"name": "Level 3: In-Context\n(Highest Frequency)", "y": 2.5, "width": 3},
    ]

    for i, level in enumerate(levels):
        rect = plt.Rectangle(
            (5 - level["width"] / 2, level["y"] - 0.8),
            level["width"],
            1.6,
            facecolor=colors[i],
            edgecolor="black",
            linewidth=2,
            alpha=0.7,
        )
        ax.add_patch(rect)
        ax.text(
            5, level["y"], level["name"], ha="center", va="center", fontsize=10, fontweight="bold"
        )

    # Draw arrows between levels
    for i in range(len(levels) - 1):
        ax.annotate(
            "",
            xy=(5, levels[i + 1]["y"] + 1),
            xytext=(5, levels[i]["y"] - 1),
            arrowprops=dict(arrowstyle="->", color="black", lw=2),
        )

    # Title
    ax.text(
        5,
        9.5,
        "Nested Learning: Multi-Level Optimization",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    # Side annotations
    ax.text(0.5, 5.5, "Knowledge\nTransfer\n↓", ha="center", va="center", fontsize=10)
    ax.text(9.5, 5.5, "Context\nCompression\n↑", ha="center", va="center", fontsize=10)

    plt.tight_layout()
    return fig


def create_demo():
    """Create the Gradio demo interface."""

    with gr.Blocks(title="Nested Learning Explorer") as demo:
        gr.Markdown("""
        # 🧠 Nested Learning Explorer

        Interactive visualization of components from Google Research's
        **"Nested Learning: The Illusion of Deep Learning Architecture"** paper.

        > *"We present a new learning paradigm, called Nested Learning (NL), that coherently
        represents a machine learning model with a set of nested, multi-level, and/or parallel
        optimization problems, each of which with its own 'context flow'."*
        """)

        with gr.Tabs():
            # Tab 1: Overview
            with gr.TabItem("📖 Overview"):
                gr.Markdown("""
                ## Key Insights from Nested Learning

                ### 1. Optimizers are Associative Memories
                Adam, SGD with Momentum, etc., are associative memory modules that compress
                gradient information.

                ### 2. Architectures are Uniform
                All neural architectures can be decomposed into feedforward networks with
                different update frequencies.

                ### 3. Pre-training is In-Context Learning
                With an ultra-large context (the entire training data).

                ### 4. Continuum Memory System
                Generalizes long-term/short-term memory with a spectrum of update frequencies.
                """)

                paradigm_plot = gr.Plot(label="Nested Learning Paradigm")

                def show_paradigm():
                    return visualize_nested_learning_paradigm()

                paradigm_btn = gr.Button("Show Nested Learning Structure")
                paradigm_btn.click(show_paradigm, outputs=paradigm_plot)

            # Tab 2: CMS Visualization
            with gr.TabItem("🔄 Continuum Memory System"):
                gr.Markdown("""
                ## Continuum Memory System (CMS)

                CMS replaces traditional MLP blocks with a chain of blocks that update
                at different frequencies:
                - **High-frequency blocks**: Fast adaptation, short-term memory
                - **Low-frequency blocks**: Slow adaptation, long-term memory

                This helps with continual learning because forgotten knowledge in
                high-frequency blocks may still exist in low-frequency blocks.
                """)

                with gr.Row():
                    cms_levels = gr.Slider(2, 6, value=4, step=1, label="Number of Levels")
                    cms_chunk = gr.Slider(4, 32, value=8, step=4, label="Base Chunk Size")
                    cms_steps = gr.Slider(32, 256, value=64, step=32, label="Total Steps")

                cms_plot = gr.Plot(label="CMS Update Schedule")
                cms_btn = gr.Button("Visualize Update Schedule")
                cms_btn.click(
                    visualize_cms_update_schedule,
                    inputs=[cms_levels, cms_chunk, cms_steps],
                    outputs=cms_plot,
                )

            # Tab 3: DGD Visualization
            with gr.TabItem("📉 Delta Gradient Descent"):
                gr.Markdown("""
                ## Delta Gradient Descent (DGD)

                Unlike standard gradient descent, DGD incorporates the current weight
                state into the update:

                ```
                W_{t+1} = W_t (I - η' x_t x_t^T) - η' ∇L(W_t; x_t)
                ```

                The first term provides **adaptive decay based on current input**,
                capturing dependencies without i.i.d. assumptions.
                """)

                with gr.Row():
                    dgd_lr = gr.Slider(0.01, 0.5, value=0.1, step=0.01, label="Learning Rate")
                    dgd_normalized = gr.Checkbox(value=True, label="Normalized Inputs")

                dgd_plot = gr.Plot(label="DGD Adaptive Decay Behavior")
                dgd_btn = gr.Button("Visualize DGD Behavior")
                dgd_btn.click(
                    lambda lr, norm: visualize_dgd_decay(lr, norm),
                    inputs=[dgd_lr, dgd_normalized],
                    outputs=dgd_plot,
                )

            # Tab 4: Memory Compression
            with gr.TabItem("💾 Memory Compression"):
                gr.Markdown("""
                ## Associative Memory as Context Compression

                In Nested Learning, both optimizers and architectures are viewed as
                associative memories that **compress their context into parameters**.

                - **Tokens** → Compressed into layer parameters
                - **Gradients** → Compressed into optimizer state
                """)

                with gr.Row():
                    mem_seq = gr.Slider(32, 128, value=64, step=16, label="Sequence Length")
                    mem_dim = gr.Slider(16, 64, value=32, step=8, label="Model Dimension")

                mem_plot = gr.Plot(label="Memory Compression Visualization")
                mem_btn = gr.Button("Visualize Compression")
                mem_btn.click(
                    lambda s, d: visualize_memory_compression(s, d),
                    inputs=[mem_seq, mem_dim],
                    outputs=mem_plot,
                )

            # Tab 5: About
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## About This Implementation

                This is a **from-scratch implementation** of the Nested Learning paper:

                > **"Nested Learning: The Illusion of Deep Learning Architecture"**
                > Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni
                > Google Research, NeurIPS 2025

                ### Implemented Components

                | Component | Description |
                |-----------|-------------|
                | **DGD** | Delta Gradient Descent optimizer |
                | **M3** | Multi-scale Momentum Muon optimizer |
                | **CMS** | Continuum Memory System |
                | **Titans** | Self-Modifying Deep Associative Memory |
                | **Hope** | Full architecture (Titans + CMS) |

                ### Part of LeCoder Project

                *"Less Code, More Reproduction"* - Building tools to help reproduce
                AI/ML research papers from scratch.

                ---

                **Repository**: [GitHub](https://github.com/yourusername/nested-learning)
                """)

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)
