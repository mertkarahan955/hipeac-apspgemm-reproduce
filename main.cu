#include <iostream>
#include <string>
#include <chrono>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cusparse.h>

#include "CSR.h"
#include "COO.h"
#include "Vector.h"

// Include GPU kernels and utilities
#include "GPU/utils.cu"
#include "GPU/SpMM.cu"
#include "GPU/GustavsonSpGEMM.cu"

using namespace std;
using namespace std::chrono;

// ========================================
// Helper: Generate random dense matrix
// ========================================

void generateRandomDense(DenseMatrix<float>& mat, size_t rows, size_t cols) {
    mat.alloc(rows, cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        mat.data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// ========================================
// Helper: Convert CSR to COO (if needed)
// ========================================

template<typename T>
void convertCSRtoCOO(const CSR<T>& csr, COO<T>& coo) {
    coo.alloc(csr.rows, csr.cols, csr.nnz);

    size_t idx = 0;
    for (size_t i = 0; i < csr.rows; ++i) {
        int row_start = csr.row_offsets[i];
        int row_end = csr.row_offsets[i + 1];

        for (int j = row_start; j < row_end; ++j) {
            coo.row_ids[idx] = i;
            coo.col_ids[idx] = csr.col_ids[j];
            coo.data[idx] = csr.data[j];
            idx++;
        }
    }
}

// ========================================
// COO to CSR conversion
// ========================================

template<typename T>
void convertCOOtoCSR(const COO<T>& coo, CSR<T>& csr) {
    csr.alloc(coo.rows, coo.cols, coo.nnz);

    // Initialize row_offsets
    for (size_t i = 0; i <= csr.rows; ++i) {
        csr.row_offsets[i] = 0;
    }

    // Count non-zeros per row
    for (size_t i = 0; i < coo.nnz; ++i) {
        csr.row_offsets[coo.row_ids[i] + 1]++;
    }

    // Cumulative sum
    for (size_t i = 1; i <= csr.rows; ++i) {
        csr.row_offsets[i] += csr.row_offsets[i - 1];
    }

    // Fill data and col_ids
    std::vector<unsigned int> row_counters(csr.rows, 0);
    for (size_t i = 0; i < coo.nnz; ++i) {
        unsigned int row = coo.row_ids[i];
        unsigned int offset = csr.row_offsets[row] + row_counters[row];
        csr.col_ids[offset] = coo.col_ids[i];
        csr.data[offset] = coo.data[i];
        row_counters[row]++;
    }
}

// ========================================
// Test SpMM
// ========================================

void testSpMM(const CSR<float>& A) {
    cout << "\n========================================" << endl;
    cout << "Testing SpMM (Sparse × Dense)" << endl;
    cout << "========================================" << endl;

    int M = A.rows;
    int K = A.cols;
    int N = 256;  // Dense matrix columns

    // Generate random dense matrix B
    DenseMatrix<float> B, Y_cpu, Y_gpu;
    generateRandomDense(B, K, N);
    Y_cpu.alloc(M, N);
    Y_gpu.alloc(M, N);

    // CPU baseline
    cout << "Running CPU SpMM..." << endl;
    auto t1 = high_resolution_clock::now();
    cpuSpMM(A, B, Y_cpu);
    auto t2 = high_resolution_clock::now();
    double cpu_time = duration_cast<microseconds>(t2 - t1).count() / 1000.0;
    cout << "  CPU time: " << cpu_time << " ms" << endl;

    // GPU SpMM
    cout << "Running GPU SpMM..." << endl;
    dCSR<float> d_A;
    dDenseMatrix<float> d_B, d_Y;

    transferCSRToDevice(A, d_A);
    transferDenseToDevice(B, d_B);
    allocateDeviceDense(d_Y, M, N);
    zeroDenseDevice(d_Y);

    cudaDeviceSynchronize();
    t1 = high_resolution_clock::now();
    gpuSpMM(d_A, d_B, d_Y);
    cudaDeviceSynchronize();
    t2 = high_resolution_clock::now();
    double gpu_time = duration_cast<microseconds>(t2 - t1).count() / 1000.0;
    cout << "  GPU time: " << gpu_time << " ms" << endl;

    transferDenseToHost(d_Y, Y_gpu);

    // Validate
    bool valid = validateSpMM(Y_cpu, Y_gpu);
    if (valid) {
        cout << "  Speedup: " << cpu_time / gpu_time << "x" << endl;
    }

    // cuSPARSE baseline
    cout << "Running cuSPARSE SpMM..." << endl;
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    dDenseMatrix<float> d_Y_cusparse;
    allocateDeviceDense(d_Y_cusparse, M, N);
    zeroDenseDevice(d_Y_cusparse);

    cudaDeviceSynchronize();
    t1 = high_resolution_clock::now();
    cusparseSpMM(d_A, d_B, d_Y_cusparse, handle);
    cudaDeviceSynchronize();
    t2 = high_resolution_clock::now();
    double cusparse_time = duration_cast<microseconds>(t2 - t1).count() / 1000.0;
    cout << "  cuSPARSE time: " << cusparse_time << " ms" << endl;
    cout << "  vs cuSPARSE: " << cusparse_time / gpu_time << "x" << endl;

    cusparseDestroy(handle);
    freeDeviceCSR(d_A);
    freeDeviceDense(d_B);
    freeDeviceDense(d_Y);
    freeDeviceDense(d_Y_cusparse);
}

// ========================================
// Test SpGEMM
// ========================================

void testSpGEMM(const CSR<float>& A, const CSR<float>& B) {
    cout << "\n========================================" << endl;
    cout << "Testing SpGEMM (Sparse × Sparse)" << endl;
    cout << "========================================" << endl;
    cout << "A: " << A.rows << " × " << A.cols << " (nnz=" << A.nnz << ")" << endl;
    cout << "B: " << B.rows << " × " << B.cols << " (nnz=" << B.nnz << ")" << endl;

    CSR<float> C_cpu, C_gpu;

    // CPU baseline
    cout << "Running CPU Gustavson SpGEMM..." << endl;
    auto t1 = high_resolution_clock::now();
    cpuGustavsonSpGEMM(A, B, C_cpu);
    auto t2 = high_resolution_clock::now();
    double cpu_time = duration_cast<microseconds>(t2 - t1).count() / 1000.0;
    cout << "  CPU time: " << cpu_time << " ms" << endl;
    cout << "  Output: " << C_cpu.rows << " × " << C_cpu.cols << " (nnz=" << C_cpu.nnz << ")" << endl;

    // GPU SpGEMM
    cout << "Running GPU Gustavson SpGEMM..." << endl;
    dCSR<float> d_A, d_B, d_C;

    transferCSRToDevice(A, d_A);
    transferCSRToDevice(B, d_B);

    cudaDeviceSynchronize();
    t1 = high_resolution_clock::now();
    gpuGustavsonSpGEMM(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    t2 = high_resolution_clock::now();
    double gpu_time = duration_cast<microseconds>(t2 - t1).count() / 1000.0;
    cout << "  GPU time: " << gpu_time << " ms" << endl;

    transferCSRToHost(d_C, C_gpu);
    cout << "  Output: " << C_gpu.rows << " × " << C_gpu.cols << " (nnz=" << C_gpu.nnz << ")" << endl;

    // Validate (basic NNZ check)
    if (C_cpu.nnz == C_gpu.nnz) {
        cout << "  ✓ NNZ match!" << endl;
        cout << "  Speedup: " << cpu_time / gpu_time << "x" << endl;
    } else {
        cout << "  ✗ NNZ mismatch: CPU=" << C_cpu.nnz << " GPU=" << C_gpu.nnz << endl;
    }

    // Detailed validation (check values)
    int errors = 0;
    for (size_t i = 0; i < C_cpu.rows && errors < 10; ++i) {
        int cpu_start = C_cpu.row_offsets[i];
        int cpu_end = C_cpu.row_offsets[i + 1];
        int gpu_start = C_gpu.row_offsets[i];
        int gpu_end = C_gpu.row_offsets[i + 1];

        if (cpu_end - cpu_start != gpu_end - gpu_start) {
            cout << "  Row " << i << " NNZ mismatch: CPU=" << (cpu_end - cpu_start)
                 << " GPU=" << (gpu_end - gpu_start) << endl;
            errors++;
            continue;
        }

        for (int j = cpu_start; j < cpu_end && errors < 10; ++j) {
            int gpu_j = gpu_start + (j - cpu_start);
            if (C_cpu.col_ids[j] != C_gpu.col_ids[gpu_j]) {
                cout << "  Row " << i << " col mismatch at offset " << (j - cpu_start) << endl;
                errors++;
            } else if (fabs(C_cpu.data[j] - C_gpu.data[gpu_j]) > 1e-4) {
                cout << "  Row " << i << " value mismatch: CPU=" << C_cpu.data[j]
                     << " GPU=" << C_gpu.data[gpu_j] << endl;
                errors++;
            }
        }
    }

    if (errors == 0) {
        cout << "  ✓ Full validation passed!" << endl;
    }

    freeDeviceCSR(d_A);
    freeDeviceCSR(d_B);
    freeDeviceCSR(d_C);
}

// ========================================
// Main
// ========================================

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <matrix.mtx> [<matrix2.mtx>]" << endl;
        cout << "  If one matrix: test SpMM (sparse × dense)" << endl;
        cout << "  If two matrices: test SpGEMM (sparse × sparse)" << endl;
        return 1;
    }

    srand(42);  // Reproducible random numbers

    cout << "ApSpGEMM GPU Porting - Test Suite" << endl;
    cout << "========================================" << endl;

    try {
        // Load matrix A
        cout << "Loading matrix A from: " << argv[1] << endl;
        COO<float> coo_A = loadMTX<float>(argv[1]);
        CSR<float> A;
        convertCOOtoCSR(coo_A, A);
        cout << "  Matrix A: " << A.rows << " × " << A.cols << " (nnz=" << A.nnz << ")" << endl;

        if (argc == 2) {
            // Test SpMM only
            testSpMM(A);
        } else {
            // Load matrix B and test SpGEMM
            cout << "Loading matrix B from: " << argv[2] << endl;
            COO<float> coo_B = loadMTX<float>(argv[2]);
            CSR<float> B;
            convertCOOtoCSR(coo_B, B);
            cout << "  Matrix B: " << B.rows << " × " << B.cols << " (nnz=" << B.nnz << ")" << endl;

            if (A.cols != B.rows) {
                cerr << "Error: Dimension mismatch for SpGEMM (A.cols != B.rows)" << endl;
                return 1;
            }

            testSpGEMM(A, B);
        }

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    cout << "\n========================================" << endl;
    cout << "All tests completed!" << endl;
    cout << "========================================" << endl;

    return 0;
}
