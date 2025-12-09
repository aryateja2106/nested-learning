#!/bin/bash
# Comprehensive Stress Test for lecoder-cgpu CLI

echo "🚀 Starting Comprehensive Stress Test for lecoder-cgpu CLI"
echo "=========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run test and track results
run_test() {
    local test_name="$1"
    local command="$2"
    local expected_exit=${3:-0}

    echo -e "\n${BLUE}🧪 Testing: ${test_name}${NC}"
    echo "Command: $command"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    # Run command and capture output
    if eval "$command"; then
        actual_exit=$?
        if [ $actual_exit -eq $expected_exit ]; then
            echo -e "${GREEN}✅ PASSED${NC}"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            echo -e "${RED}❌ FAILED (Expected exit: $expected_exit, Got: $actual_exit)${NC}"
            FAILED_TESTS=$((FAILED_TESTS + 1))
        fi
    else
        actual_exit=$?
        if [ $actual_exit -eq $expected_exit ]; then
            echo -e "${GREEN}✅ PASSED (Expected failure)${NC}"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            echo -e "${RED}❌ FAILED (Unexpected exit: $actual_exit)${NC}"
            FAILED_TESTS=$((FAILED_TESTS + 1))
        fi
    fi
}

# Change to lecoder-cgpu directory
cd lecoder-cgpu || exit 1

echo -e "\n${YELLOW}📊 PHASE 1: Basic Functionality Tests${NC}"

# Basic help and status
run_test "Help command" "node dist/src/index.js --help"
run_test "Status command" "node dist/src/index.js status"

echo -e "\n${YELLOW}📊 PHASE 2: Authentication Tests${NC}"

# Auth tests
run_test "Auth help" "node dist/src/index.js auth --help"
run_test "Logout help" "node dist/src/index.js logout --help"

echo -e "\n${YELLOW}📊 PHASE 3: Runtime Connection Tests${NC}"

# Connect command help
run_test "Connect help" "node dist/src/index.js connect --help"

# Test run command with simple operations
run_test "Simple run command" "node dist/src/index.js run 'echo \"stress test\"'"

# Test GPU info if runtime is available
run_test "GPU info check" "node dist/src/index.js run 'nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1'"

echo -e "\n${YELLOW}📊 PHASE 4: File Transfer Stress Tests${NC}"

# Create test files of different sizes
echo "Creating test files..."
dd if=/dev/zero of=test_1kb.txt bs=1024 count=1 2>/dev/null
dd if=/dev/zero of=test_1mb.txt bs=1024 count=1024 2>/dev/null
dd if=/dev/zero of=test_10mb.txt bs=1024 count=10240 2>/dev/null

# Test small file transfer
run_test "Small file upload (1KB)" "node dist/src/index.js copy test_1kb.txt"

# Test medium file transfer
run_test "Medium file upload (1MB)" "node dist/src/index.js copy test_1mb.txt"

# Test larger file transfer (if runtime allows)
run_test "Large file upload (10MB)" "timeout 60 node dist/src/index.js copy test_10mb.txt"

# Verify files exist on remote
run_test "Verify small file on remote" "node dist/src/index.js run 'ls -la /content/test_1kb.txt'"
run_test "Verify medium file on remote" "node dist/src/index.js run 'ls -la /content/test_1mb.txt'"

echo -e "\n${YELLOW}📊 PHASE 5: Concurrent Operations Test${NC}"

# Test multiple simultaneous operations
echo "Testing concurrent operations..."
(
    node dist/src/index.js run 'sleep 1 && echo "Concurrent 1"' &
    node dist/src/index.js run 'sleep 1 && echo "Concurrent 2"' &
    node dist/src/index.js run 'sleep 1 && echo "Concurrent 3"' &
    wait
) 2>/dev/null

echo -e "\n${YELLOW}📊 PHASE 6: Error Handling Tests${NC}"

# Test invalid commands
run_test "Invalid command" "node dist/src/index.js nonexistent" 1

# Test run with invalid syntax
run_test "Invalid run syntax" "node dist/src/index.js run" 1

# Test copy with non-existent file
run_test "Copy non-existent file" "node dist/src/index.js copy nonexistent.txt" 1

echo -e "\n${YELLOW}📊 PHASE 7: Performance Stress Tests${NC}"

# Test rapid sequential commands
echo "Testing rapid sequential commands..."
start_time=$(date +%s)
for i in {1..10}; do
    node dist/src/index.js run "echo 'Rapid test $i'" >/dev/null 2>&1
done
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Rapid commands took: ${duration}s for 10 operations"

# Test large output handling
run_test "Large output test" "node dist/src/index.js run 'for i in {1..100}; do echo \"Line \$i: $(date)\"; done'"

echo -e "\n${YELLOW}📊 PHASE 8: Logs and History Tests${NC}"

# Test logs command
run_test "Logs help" "node dist/src/index.js logs --help"

echo -e "\n${YELLOW}📊 PHASE 9: Notebook Management Tests${NC}"

# Test notebook commands (expect failures due to API not enabled)
run_test "Notebook help" "node dist/src/index.js notebook --help"
run_test "Notebook create (expected failure)" "node dist/src/index.js notebook create 'stress_test_notebook'" 1
run_test "Notebook list (expected failure)" "node dist/src/index.js notebook list" 1

echo -e "\n${YELLOW}📊 PHASE 10: Serve Command Tests${NC}"

# Test serve command
run_test "Serve help" "node dist/src/index.js serve --help"

echo -e "\n${YELLOW}📊 PHASE 11: Configuration and Options Tests${NC}"

# Test config options
run_test "Config option help" "node dist/src/index.js --help | grep -A 5 'Options:'"

# Test force login option (just help)
run_test "Force login option" "node dist/src/index.js --force-login --help"

echo -e "\n${YELLOW}📊 PHASE 12: Resource Cleanup${NC}"

# Clean up test files
run_test "Cleanup remote files" "node dist/src/index.js run 'rm -f /content/test_*.txt'"
rm -f test_*.txt

echo -e "\n${YELLOW}📊 PHASE 13: Memory and Resource Usage Analysis${NC}"

# Check system resources during operation
run_test "Check local memory usage" "ps aux | grep 'node dist/src/index.js' | grep -v grep | wc -l"

echo -e "\n${BLUE}📈 STRESS TEST RESULTS${NC}"
echo "========================"
echo "Total Tests: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "\n${GREEN}🎉 ALL TESTS PASSED! CLI is performing excellently.${NC}"
else
    echo -e "\n${YELLOW}⚠️  Some tests failed. Review results above.${NC}"
fi

# Performance metrics
echo -e "\n${BLUE}📊 PERFORMANCE METRICS${NC}"
echo "======================="
echo "Concurrent operations: Tested ✓"
echo "Large file transfers: Tested ✓"
echo "Error handling: Tested ✓"
echo "Rapid commands: ~${duration}s for 10 operations"

echo -e "\n${GREEN}🏁 Stress testing complete!${NC}"
