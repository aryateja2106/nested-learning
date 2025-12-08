#!/usr/bin/env bash
# Enterprise Continual Learning Experiment via LeCoder cGPU CLI
# ==============================================================
#
# This script demonstrates the full capabilities of LeCoder cGPU CLI:
# - Notebook management (create, open)
# - File transfer (copy)
# - Remote execution (run with kernel mode)
# - Session management (list, switch)
# - Execution history (logs)
# - Status monitoring (status)
#
# Usage:
#   ./run_lecoder_experiment.sh [mode] [options]
#
# Modes:
#   quick      - Quick GPU test and benchmark
#   train      - Full training experiment
#   benchmark  - CUDA performance benchmark only
#   full       - Complete workflow showcase (default)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect which cgpu binary to use
if command -v lecoder-cgpu &> /dev/null; then
    CGPU_BIN="lecoder-cgpu"
elif command -v cgpu &> /dev/null; then
    CGPU_BIN="cgpu"
    echo -e "${YELLOW}⚠️  Warning: Using legacy 'cgpu' command. Consider installing 'lecoder-cgpu' instead.${NC}"
else
    echo -e "${RED}❌ Error: Neither 'lecoder-cgpu' nor 'cgpu' found in PATH.${NC}"
    echo "Please install lecoder-cgpu:"
    echo "  npm install -g lecoder-cgpu"
    echo "Or from source:"
    echo "  cd lecoder-cgpu && npm install && npm link"
    exit 1
fi

echo -e "${BLUE}Using CLI: $CGPU_BIN${NC}\n"

# Configuration
REPO_NAME="lecoder-nested-learning"
EXPERIMENT_NAME="HOPE-Enterprise-Experiment"
REMOTE_DIR="/content/${REPO_NAME}"
ARCHIVE="/tmp/${REPO_NAME}.tar.gz"
MODE="${1:-full}"

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

# Phase 1: Setup and Authentication
phase1_setup() {
    log_info "Phase 1: Setup and Authentication"
    echo "─────────────────────────────────────────"
    
    # Check authentication status
    log_info "Checking authentication status..."
    
    # Get status as JSON for parsing
    STATUS_JSON=$($CGPU_BIN status --json 2>/dev/null)
    
    if [ -z "$STATUS_JSON" ]; then
        log_error "Could not get status. Please run: $CGPU_BIN auth"
        exit 1
    fi
    
    # Check if authenticated (handle both boolean true and string "true")
    AUTHENTICATED=$(echo "$STATUS_JSON" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    auth = data.get('authenticated', False)
    print('true' if auth else 'false')
except:
    print('false')
" 2>/dev/null || echo "false")
    
    if [ "$AUTHENTICATED" != "true" ]; then
        log_error "Authentication failed. Please run: $CGPU_BIN auth"
        exit 1
    fi
    
    log_success "Authenticated successfully"
    
    # Show eligible GPUs
    ELIGIBLE_GPUS=$(echo "$STATUS_JSON" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    gpus = data.get('eligibleGpus', [])
    print(', '.join(gpus) if gpus else 'None')
except:
    print('Unknown')
" 2>/dev/null || echo "Unknown")
    log_info "Eligible GPUs: $ELIGIBLE_GPUS"
    echo ""
}

# Phase 2: Notebook Management
phase2_notebook() {
    log_info "Phase 2: Notebook Management"
    echo "─────────────────────────────────────────"
    
    # Create experiment notebook
    log_info "Creating experiment notebook: $EXPERIMENT_NAME"
    NOTEBOOK_JSON=$($CGPU_BIN notebook create "$EXPERIMENT_NAME" --template gpu --json 2>/dev/null || echo "{}")
    
    NOTEBOOK_ID=$(echo "$NOTEBOOK_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', ''))" 2>/dev/null || echo "")
    
    if [ -z "$NOTEBOOK_ID" ]; then
        log_warning "Could not create notebook (may already exist). Continuing..."
    else
        log_success "Notebook created: $NOTEBOOK_ID"
        NOTEBOOK_LINK=$(echo "$NOTEBOOK_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin).get('webViewLink', ''))" 2>/dev/null || echo "")
        if [ -n "$NOTEBOOK_LINK" ]; then
            log_info "View notebook: $NOTEBOOK_LINK"
        fi
    fi
    
    echo ""
}

# Phase 3: Session Management
phase3_sessions() {
    log_info "Phase 3: Session Management"
    echo "─────────────────────────────────────────"
    
    # List active sessions
    log_info "Listing active sessions..."
    SESSIONS_JSON=$($CGPU_BIN sessions list --json 2>/dev/null || echo "[]")
    SESSION_COUNT=$(echo "$SESSIONS_JSON" | python3 -c "import sys, json; sessions = json.load(sys.stdin); print(len(sessions) if isinstance(sessions, list) else 0)" 2>/dev/null || echo "0")
    
    log_info "Active sessions: $SESSION_COUNT"
    
    # Show session stats
    log_info "Session statistics:"
    $CGPU_BIN sessions list --stats 2>/dev/null || log_warning "Could not fetch session stats"
    
    echo ""
}

# Phase 4: File Transfer
phase4_file_transfer() {
    log_info "Phase 4: File Transfer"
    echo "─────────────────────────────────────────"
    
    # Pack repository
    log_info "Packing repository..."
    tar -czf "$ARCHIVE" \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.venv' \
        --exclude='venv' \
        --exclude='node_modules' \
        --exclude='dist' \
        --exclude='*.egg-info' \
        . > /dev/null 2>&1
    
    ARCHIVE_SIZE=$(du -h "$ARCHIVE" | cut -f1)
    log_success "Repository packed: $ARCHIVE_SIZE"
    
    # Upload archive
    log_info "Uploading to Colab via $CGPU_BIN copy..."
    $CGPU_BIN copy "$ARCHIVE" "/content/${REPO_NAME}.tar.gz" || {
        log_error "File upload failed"
        exit 1
    }
    log_success "Archive uploaded"
    
    # Extract on remote
    log_info "Extracting repository on Colab..."
    $CGPU_BIN run "rm -rf ${REMOTE_DIR} && mkdir -p ${REMOTE_DIR} && tar -xzf /content/${REPO_NAME}.tar.gz -C ${REMOTE_DIR}" || {
        log_error "Extraction failed"
        exit 1
    }
    log_success "Repository extracted"
    
    echo ""
}

# Phase 5: Environment Setup
phase5_environment() {
    log_info "Phase 5: Environment Setup"
    echo "─────────────────────────────────────────"
    
    # Install dependencies
    log_info "Installing dependencies..."
    $CGPU_BIN run "cd ${REMOTE_DIR} && pip install -q -r requirements.txt" || {
        log_error "Dependency installation failed"
        exit 1
    }
    log_success "Dependencies installed"
    
    echo ""
}

# Phase 6: GPU Verification
phase6_gpu_verify() {
    log_info "Phase 6: GPU Verification"
    echo "─────────────────────────────────────────"
    
    # Check GPU using terminal mode (more reliable than kernel mode)
    log_info "Verifying GPU availability..."
    GPU_CHECK_OUTPUT=$($CGPU_BIN run "python3 -c \"import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')\"" 2>/dev/null | grep -E "(CUDA Available|GPU:)" || echo "")
    
    if echo "$GPU_CHECK_OUTPUT" | grep -q "CUDA Available: True"; then
        DEVICE_NAME=$(echo "$GPU_CHECK_OUTPUT" | grep "GPU:" | sed 's/GPU: //' | head -1)
        log_success "GPU available: $DEVICE_NAME"
    else
        log_warning "GPU not available, will use CPU"
    fi
    
    echo ""
}

# Phase 7: Quick Test
phase7_quick_test() {
    log_info "Phase 7: Quick GPU Test"
    echo "─────────────────────────────────────────"
    
    # Use terminal mode with heredoc for multi-line Python code (more reliable)
    log_info "Running quick test..."
    $CGPU_BIN run "cd ${REMOTE_DIR} && python3 << 'PYEOF'
import torch
import sys
sys.path.insert(0, '.')
from src.models.hope import Hope, HopeConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

config = HopeConfig(
    d_model=64, d_hidden=256, d_key=16, d_value=16, num_heads=4,
    num_layers=1, vocab_size=256, max_seq_len=128,
    cms_num_levels=2, cms_base_chunk_size=4,
)

model = Hope(config).to(device)
print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)
labels = torch.randint(0, config.vocab_size, (2, 16), device=device)

with torch.no_grad():
    output = model(input_ids, labels=labels)
    loss_val = output['loss'].item()
    print(f'✓ Forward pass successful!')
    print(f'  Loss: {loss_val:.4f}')
    print(f'  Logits shape: {output[\"logits\"].shape}')
    
if torch.cuda.is_available():
    print(f'  GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB')
PYEOF
" || {
        log_warning "Quick test failed (this is non-critical, continuing...)"
        return 0
    }
    log_success "Quick test completed"
    
    echo ""
}

# Phase 8: CUDA Benchmark
phase8_benchmark() {
    log_info "Phase 8: CUDA Performance Benchmark"
    echo "─────────────────────────────────────────"
    
    log_info "Running CUDA benchmark..."
    BENCHMARK_OUTPUT=$($CGPU_BIN run "cd ${REMOTE_DIR} && python3 << 'PYEOF'
import torch
import sys
import json
sys.path.insert(0, '.')
from src.experiments.cuda_kernels import benchmark_cuda_operations

if torch.cuda.is_available():
    results = benchmark_cuda_operations(
        batch_size=32,
        seq_len=512,
        d_model=512,
        num_iterations=100
    )
    print(json.dumps(results, indent=2))
else:
    print(json.dumps({'error': 'CUDA not available'}))
PYEOF
" 2>/dev/null || echo "")
    
    if echo "$BENCHMARK_OUTPUT" | grep -q "operations_per_second"; then
        log_success "Benchmark completed"
        echo "$BENCHMARK_OUTPUT" | grep -A 10 "operations_per_second" || echo "$BENCHMARK_OUTPUT"
    else
        log_warning "Benchmark not available (CUDA may not be available or still initializing)"
    fi
    
    echo ""
}

# Phase 9: Full Training Experiment
phase9_training() {
    log_info "Phase 9: Full Training Experiment"
    echo "─────────────────────────────────────────"
    
    # $1 is config, $2 is steps (when called with arguments)
    CONFIG="${1:-a100}"
    STEPS="${2:-1000}"
    
    log_info "Running enterprise pipeline experiment..."
    log_info "Config: $CONFIG, Steps: $STEPS"
    log_info "Starting training (this may take several minutes)..."
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Training in progress... (Watch for progress updates below)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    TRAIN_CMD="cd ${REMOTE_DIR} && python -m src.experiments.enterprise_pipeline --config $CONFIG --steps $STEPS --output-json"
    
    # Run training and capture output
    TRAIN_OUTPUT=$($CGPU_BIN run --verbose "$TRAIN_CMD" 2>&1)
    TRAIN_EXIT_CODE=$?
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        log_success "Training completed successfully!"
        
        # Extract and display key metrics if available
        if echo "$TRAIN_OUTPUT" | grep -q "Final Results"; then
            echo ""
            log_info "Final Training Results:"
            echo "$TRAIN_OUTPUT" | grep -A 20 "Final Results" | head -25
        fi
        
        # Show JSON output if available
        if echo "$TRAIN_OUTPUT" | grep -q "\"status\""; then
            echo ""
            log_info "Training Summary (JSON):"
            echo "$TRAIN_OUTPUT" | grep -E "\"(status|total_steps|final_loss|throughput)\"" | head -10
        fi
    else
        log_error "Training failed with exit code: $TRAIN_EXIT_CODE"
        echo ""
        log_info "Last 20 lines of output:"
        echo "$TRAIN_OUTPUT" | tail -20
        exit 1
    fi
    
    echo ""
}

# Phase 10: Execution History
phase10_history() {
    log_info "Phase 10: Execution History"
    echo "─────────────────────────────────────────"
    
    log_info "Recent execution history:"
    $CGPU_BIN logs --limit 5 || log_warning "Could not fetch execution history"
    
    log_info "Execution statistics:"
    $CGPU_BIN logs --stats || log_warning "Could not fetch statistics"
    
    echo ""
}

# Phase 11: Final Status
phase11_status() {
    log_info "Phase 11: Final Status Check"
    echo "─────────────────────────────────────────"
    
    log_info "Runtime status:"
    $CGPU_BIN status || log_warning "Could not fetch status"
    
    echo ""
}

# Main execution based on mode
case "$MODE" in
    quick)
        log_info "Running QUICK mode: GPU test and benchmark"
        phase1_setup
        phase6_gpu_verify
        phase7_quick_test
        phase8_benchmark
        phase10_history
        ;;
    
    train)
        log_info "Running TRAIN mode: Full training experiment"
        CONFIG="${2:-a100}"
        STEPS="${3:-1000}"
        phase1_setup
        phase4_file_transfer
        phase5_environment
        phase6_gpu_verify
        phase9_training "$CONFIG" "$STEPS"
        phase10_history
        ;;
    
    benchmark)
        log_info "Running BENCHMARK mode: CUDA performance only"
        phase1_setup
        phase4_file_transfer
        phase5_environment
        phase6_gpu_verify
        phase8_benchmark
        ;;
    
    full|*)
        log_info "Running FULL mode: Complete workflow showcase"
        phase1_setup
        phase2_notebook
        phase3_sessions
        phase4_file_transfer
        phase5_environment
        phase6_gpu_verify
        phase7_quick_test
        phase8_benchmark
        
        # Ask if user wants to run full training
        echo ""
        read -p "Run full training experiment? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            phase9_training "a100" "500"
        fi
        
        phase10_history
        phase11_status
        ;;
esac

log_success "Experiment workflow completed!"
log_info "Colab workspace: ${REMOTE_DIR}"
log_info "To connect interactively: $CGPU_BIN connect"

# Cleanup local archive
rm -f "$ARCHIVE"

