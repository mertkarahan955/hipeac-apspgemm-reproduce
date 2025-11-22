#!/bin/bash

# Comprehensive SpGEMM Benchmark Script
# Calculates GFlops like in the ApSpGEMM paper

echo "========================================================"
echo "   ApSpGEMM Comprehensive Benchmark Suite"
echo "   Computing GFlops for SpGEMM Performance Analysis"
echo "========================================================"
echo ""

BUILD_DIR="build"
MATRIX_DIR="benchmark_matrices"
RESULTS_FILE="benchmark_results.csv"

# Check if executable exists
if [ ! -f "$BUILD_DIR/main" ]; then
    echo "Error: Executable not found. Run 'make' first."
    exit 1
fi

# Create results header
echo "Matrix,Rows,Cols,Input_NNZ,Output_NNZ,CPU_ms,GPU_ms,Speedup,GFlops_CPU,GFlops_GPU" > $RESULTS_FILE

# Function to calculate GFlops
# For SpGEMM C = A × A:
# FLOPs ≈ 2 × (number of multiply-add operations)
# For each output element: sum of (A[i,k] * A[k,j]) requires 2 FLOPs per product
calculate_gflops() {
    local nnz_A=$1
    local nnz_C=$2
    local time_ms=$3

    # Estimated FLOPs: For SpGEMM, each output NNZ requires approximately
    # (nnz_per_row_A) multiply-adds = 2 × nnz_per_row_A FLOPs
    # Total ≈ 2 × nnz_C × sqrt(nnz_A)
    # Simplified: 2 × nnz_A × nnz_C / rows (approximation)

    # More accurate: 2 × nnz_C (since each output accumulates multiple products)
    # Conservative estimate: 2 × nnz_A (input operations)
    local flops=$(echo "2 * $nnz_C" | bc)

    # GFlops = FLOPs / (time_ms × 10^6)
    local gflops=$(echo "scale=2; $flops / ($time_ms * 1000000)" | bc)
    echo "$gflops"
}

test_matrix() {
    local matrix_file=$1
    local matrix_name=$(basename "$matrix_file" .mtx)

    echo "---------------------------------------------------"
    echo "Testing: $matrix_name"
    echo "---------------------------------------------------"

    # Run test and capture output
    output=$(cd $BUILD_DIR && ./main "../$matrix_file" "../$matrix_file" 2>&1)

    # Extract values using grep and awk
    rows=$(echo "$output" | grep "Matrix A:" | awk '{print $3}')
    cols=$(echo "$output" | grep "Matrix A:" | awk '{print $5}')
    nnz_A=$(echo "$output" | grep "Matrix A:" | grep -o 'nnz=[0-9]*' | cut -d= -f2)
    nnz_C=$(echo "$output" | grep "Output:" | head -1 | grep -o 'nnz=[0-9]*' | cut -d= -f2)
    cpu_time=$(echo "$output" | grep "CPU time:" | awk '{print $3}')
    gpu_time=$(echo "$output" | grep "GPU time:" | awk '{print $3}')
    speedup=$(echo "$output" | grep "Speedup:" | awk '{print $2}' | tr -d 'x')

    # Check if extraction was successful
    if [ -z "$rows" ] || [ -z "$cpu_time" ] || [ -z "$gpu_time" ]; then
        echo "  ⚠ Test failed or timed out"
        return
    fi

    # Calculate GFlops
    gflops_cpu=$(calculate_gflops $nnz_A $nnz_C $cpu_time)
    gflops_gpu=$(calculate_gflops $nnz_A $nnz_C $gpu_time)

    # Print results
    echo "  Dimensions: ${rows} × ${cols}"
    echo "  Input NNZ:  $nnz_A"
    echo "  Output NNZ: $nnz_C"
    echo "  CPU Time:   ${cpu_time} ms → ${gflops_cpu} GFlops"
    echo "  GPU Time:   ${gpu_time} ms → ${gflops_gpu} GFlops"
    echo "  Speedup:    ${speedup}×"
    echo ""

    # Append to CSV
    echo "$matrix_name,$rows,$cols,$nnz_A,$nnz_C,$cpu_time,$gpu_time,$speedup,$gflops_cpu,$gflops_gpu" >> $RESULTS_FILE
}

# Test all synthetic matrices
echo "=========================================="
echo " Testing Synthetic Matrices"
echo "=========================================="
echo ""

for matrix in $MATRIX_DIR/synthetic_*.mtx; do
    if [ -f "$matrix" ]; then
        test_matrix "$matrix"
    fi
done

# Test real-world matrices (if any with coordinate format)
echo "=========================================="
echo " Testing Real-World Matrices"
echo "=========================================="
echo ""

for matrix in $MATRIX_DIR/*.mtx; do
    name=$(basename "$matrix")
    # Skip synthetic and known dense matrices
    if [[ ! "$name" =~ synthetic ]] && [[ ! "$name" =~ cit-Patents ]]; then
        # Check if it's coordinate format
        format=$(head -1 "$matrix" | grep -o "coordinate")
        if [ ! -z "$format" ]; then
            test_matrix "$matrix"
        else
            echo "Skipping $name (not coordinate format)"
        fi
    fi
done

echo "=========================================="
echo " Benchmark Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""

# Display summary table
if [ -f "$RESULTS_FILE" ]; then
    echo "Summary Table:"
    echo "----------------------------------------"
    column -t -s, $RESULTS_FILE
    echo ""

    # Calculate average GFlops
    avg_gpu_gflops=$(tail -n +2 $RESULTS_FILE | awk -F, '{sum+=$10; count++} END {if(count>0) print sum/count; else print 0}')
    echo "Average GPU GFlops: $avg_gpu_gflops"
fi

echo ""
echo "To compare with ApSpGEMM paper results:"
echo "  - Our method (paper): 58.62 GFlops average, 197.54 GFlops peak"
echo "  - cuSPARSE (paper): 25.42 GFlops average, 65.23 GFlops peak"
echo "  - Our GPU implementation: Check GFlops_GPU column above"
echo ""
