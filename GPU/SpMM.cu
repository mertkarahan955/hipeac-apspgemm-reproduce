#ifndef GPU_SPMM_CU
#define GPU_SPMM_CU

#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <cmath>
#include "CSR.h"
#include "Vector.h"

// ========================================
// SpMM Kernel: Y = A × B
// A: sparse CSR matrix (M × K)
// B: dense matrix (K × N)
// Y: dense matrix (M × N)
// ========================================

// Simple kernel: 1 thread per output row
__global__ void spmm_kernel_simple(
    const float* A_data,
    const unsigned int* A_col_ids,
    const unsigned int* A_row_offsets,
    const float* B_data,
    float* Y_data,
    int M,  // A.rows = Y.rows
    int K,  // A.cols = B.rows
    int N)  // B.cols = Y.cols
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        int row_start = A_row_offsets[row];
        int row_end = A_row_offsets[row + 1];

        // Compute all columns of output row
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;

            // Iterate over non-zeros in row of A
            for (int idx = row_start; idx < row_end; ++idx) {
                int a_col = A_col_ids[idx];
                float a_val = A_data[idx];

                // Access B[a_col, j] (row-major)
                float b_val = B_data[a_col * N + j];

                sum += a_val * b_val;
            }

            // Write Y[row, j]
            Y_data[row * N + j] = sum;
        }
    }
}

// Optimized kernel with shared memory for B columns (for smaller N)
__global__ void spmm_kernel_shared(
    const float* A_data,
    const unsigned int* A_col_ids,
    const unsigned int* A_row_offsets,
    const float* B_data,
    float* Y_data,
    int M,
    int K,
    int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory for a chunk of B columns
    extern __shared__ float shared_B[];

    if (row < M) {
        int row_start = A_row_offsets[row];
        int row_end = A_row_offsets[row + 1];

        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;

            for (int idx = row_start; idx < row_end; ++idx) {
                int a_col = A_col_ids[idx];
                float a_val = A_data[idx];
                float b_val = B_data[a_col * N + j];

                sum += a_val * b_val;
            }

            Y_data[row * N + j] = sum;
        }
    }
}

// ========================================
// Host wrapper function
// ========================================

void gpuSpMM(
    const dCSR<float>& d_A,
    const dDenseMatrix<float>& d_B,
    dDenseMatrix<float>& d_Y)
{
    int M = d_A.rows;
    int K = d_A.cols;
    int N = d_B.cols;

    // Verify dimensions
    if (d_B.rows != static_cast<size_t>(K)) {
        throw std::runtime_error("SpMM dimension mismatch: A.cols != B.rows");
    }

    // Launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    spmm_kernel_simple<<<blocksPerGrid, threadsPerBlock>>>(
        d_A.data, d_A.col_ids, d_A.row_offsets,
        d_B.data, d_Y.data,
        M, K, N
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("SpMM kernel launch failed: ") +
                                cudaGetErrorString(err));
    }

    // Synchronize
    cudaDeviceSynchronize();
}

// ========================================
// cuSPARSE baseline for comparison
// ========================================

void cusparseSpMM(
    const dCSR<float>& d_A,
    const dDenseMatrix<float>& d_B,
    dDenseMatrix<float>& d_Y,
    cusparseHandle_t handle)
{
    int M = d_A.rows;
    int K = d_A.cols;
    int N = d_B.cols;

    float alpha = 1.0f;
    float beta = 0.0f;

    // Create sparse matrix descriptor
    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, M, K, d_A.nnz,
                     (void*)d_A.row_offsets, (void*)d_A.col_ids, (void*)d_A.data,
                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // Create dense matrix descriptors
    cusparseDnMatDescr_t matB, matC;
    cusparseCreateDnMat(&matB, K, N, N, (void*)d_B.data, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matC, M, N, N, (void*)d_Y.data, CUDA_R_32F, CUSPARSE_ORDER_ROW);

    // Allocate workspace
    size_t bufferSize = 0;
    cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize
    );

    void* dBuffer = nullptr;
    if (bufferSize > 0) {
        cudaMalloc(&dBuffer, bufferSize);
    }

    // Perform SpMM
    cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
        dBuffer
    );

    cudaDeviceSynchronize();

    // Cleanup
    if (dBuffer) cudaFree(dBuffer);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
}

// ========================================
// CPU Reference Implementation
// ========================================

void cpuSpMM(
    const CSR<float>& A,
    const DenseMatrix<float>& B,
    DenseMatrix<float>& Y)
{
    int M = A.rows;
    int N = B.cols;

    // Initialize Y to zero
    for (size_t i = 0; i < Y.rows * Y.cols; ++i) {
        Y.data[i] = 0.0f;
    }

    // Y = A × B
    for (size_t i = 0; i < A.rows; ++i) {
        int row_start = A.row_offsets[i];
        int row_end = A.row_offsets[i + 1];

        for (int idx = row_start; idx < row_end; ++idx) {
            int a_col = A.col_ids[idx];
            float a_val = A.data[idx];

            for (size_t j = 0; j < B.cols; ++j) {
                Y(i, j) += a_val * B.data[a_col * B.cols + j];
            }
        }
    }
}

// ========================================
// Validation helper
// ========================================

bool validateSpMM(
    const DenseMatrix<float>& Y_cpu,
    const DenseMatrix<float>& Y_gpu,
    float tolerance = 1e-4f)
{
    if (Y_cpu.rows != Y_gpu.rows || Y_cpu.cols != Y_gpu.cols) {
        std::cerr << "Dimension mismatch!" << std::endl;
        return false;
    }

    int errors = 0;
    float max_error = 0.0f;

    for (size_t i = 0; i < Y_cpu.rows; ++i) {
        for (size_t j = 0; j < Y_cpu.cols; ++j) {
            float cpu_val = Y_cpu.data[i * Y_cpu.cols + j];
            float gpu_val = Y_gpu.data[i * Y_gpu.cols + j];
            float error = std::abs(cpu_val - gpu_val);

            if (error > tolerance) {
                if (errors < 10) {  // Print first 10 errors
                    std::cerr << "Mismatch at (" << i << "," << j << "): "
                             << "CPU=" << cpu_val << " GPU=" << gpu_val
                             << " error=" << error << std::endl;
                }
                errors++;
                max_error = std::max(max_error, error);
            }
        }
    }

    if (errors > 0) {
        std::cerr << "Total errors: " << errors << " / " << (Y_cpu.rows * Y_cpu.cols)
                  << " (max error: " << max_error << ")" << std::endl;
        return false;
    }

    std::cout << "✓ SpMM validation passed!" << std::endl;
    return true;
}

#endif // GPU_SPMM_CU
