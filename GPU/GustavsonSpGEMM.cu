#ifndef GPU_GUSTAVSON_SPGEMM_CU
#define GPU_GUSTAVSON_SPGEMM_CU

#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include "CSR.h"

// ========================================
// Gustavson's SpGEMM on GPU
// Two-phase approach:
// Phase 1: Symbolic - count NNZ per row
// Phase 2: Numeric - compute actual values
// ========================================

#define HASH_SCALE 4  // Hash table size multiplier (increased for better performance)
#define EMPTY_KEY 0xFFFFFFFF

// Simple hash function
__device__ inline unsigned int hash_func(unsigned int key, unsigned int size) {
    return key % size;
}

// ========================================
// Phase 1: Symbolic - Count NNZ per row
// ========================================

__global__ void symbolic_kernel(
    const unsigned int* A_row_offsets,
    const unsigned int* A_col_ids,
    const unsigned int* B_row_offsets,
    const unsigned int* B_col_ids,
    unsigned int* C_row_nnz,  // Output: NNZ count per row
    unsigned int* hash_table,  // Workspace
    int M,
    int hash_size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        // Local hash table offset for this row
        unsigned int* local_hash = hash_table + row * hash_size;

        // Initialize hash table
        for (int i = 0; i < hash_size; ++i) {
            local_hash[i] = EMPTY_KEY;
        }

        int nnz_count = 0;
        int row_start_A = A_row_offsets[row];
        int row_end_A = A_row_offsets[row + 1];

        // For each non-zero in row of A
        for (int idx_A = row_start_A; idx_A < row_end_A; ++idx_A) {
            int col_A = A_col_ids[idx_A];

            // Get corresponding row in B
            int row_start_B = B_row_offsets[col_A];
            int row_end_B = B_row_offsets[col_A + 1];

            // For each non-zero in row of B
            for (int idx_B = row_start_B; idx_B < row_end_B; ++idx_B) {
                int col_B = B_col_ids[idx_B];

                // Insert col_B into hash table if not present
                unsigned int pos = hash_func(col_B, hash_size);

                // Linear probing
                while (true) {
                    unsigned int existing = local_hash[pos];

                    if (existing == col_B) {
                        // Already inserted
                        break;
                    } else if (existing == EMPTY_KEY) {
                        // Insert new column
                        local_hash[pos] = col_B;
                        nnz_count++;
                        break;
                    }

                    // Collision: try next slot
                    pos = (pos + 1) % hash_size;
                }
            }
        }

        C_row_nnz[row] = nnz_count;
    }
}

// ========================================
// Phase 2: Numeric - Compute values
// ========================================

__global__ void numeric_kernel(
    const float* A_data,
    const unsigned int* A_row_offsets,
    const unsigned int* A_col_ids,
    const float* B_data,
    const unsigned int* B_row_offsets,
    const unsigned int* B_col_ids,
    const unsigned int* C_row_offsets,  // Computed from symbolic phase
    unsigned int* C_col_ids,
    float* C_data,
    unsigned int* hash_table_keys,  // Workspace: column indices
    float* hash_table_vals,         // Workspace: accumulated values
    int M,
    int hash_size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        // Local hash table offset
        unsigned int* local_keys = hash_table_keys + row * hash_size;
        float* local_vals = hash_table_vals + row * hash_size;

        // Initialize hash table
        for (int i = 0; i < hash_size; ++i) {
            local_keys[i] = EMPTY_KEY;
            local_vals[i] = 0.0f;
        }

        int row_start_A = A_row_offsets[row];
        int row_end_A = A_row_offsets[row + 1];

        // Accumulate: C[row, :] = A[row, :] × B
        for (int idx_A = row_start_A; idx_A < row_end_A; ++idx_A) {
            int col_A = A_col_ids[idx_A];
            float val_A = A_data[idx_A];

            int row_start_B = B_row_offsets[col_A];
            int row_end_B = B_row_offsets[col_A + 1];

            for (int idx_B = row_start_B; idx_B < row_end_B; ++idx_B) {
                int col_B = B_col_ids[idx_B];
                float val_B = B_data[idx_B];

                float product = val_A * val_B;

                // Insert/accumulate in hash table
                unsigned int pos = hash_func(col_B, hash_size);

                while (true) {
                    unsigned int existing_key = local_keys[pos];

                    if (existing_key == col_B) {
                        // Accumulate
                        local_vals[pos] += product;
                        break;
                    } else if (existing_key == EMPTY_KEY) {
                        // Insert new
                        local_keys[pos] = col_B;
                        local_vals[pos] = product;
                        break;
                    }

                    pos = (pos + 1) % hash_size;
                }
            }
        }

        // Write results to CSR output (unsorted)
        int output_start = C_row_offsets[row];
        int output_idx = output_start;

        for (int i = 0; i < hash_size; ++i) {
            if (local_keys[i] != EMPTY_KEY) {
                C_col_ids[output_idx] = local_keys[i];
                C_data[output_idx] = local_vals[i];
                output_idx++;
            }
        }
    }
}

// ========================================
// Sort columns within each row (CSR standard)
// ========================================

__global__ void sort_rows_kernel(
    unsigned int* C_row_offsets,
    unsigned int* C_col_ids,
    float* C_data,
    int M)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        int row_start = C_row_offsets[row];
        int row_end = C_row_offsets[row + 1];
        int row_nnz = row_end - row_start;

        // Simple bubble sort (ok for small row NNZ)
        // For large NNZ, use thrust::sort or merge sort
        for (int i = 0; i < row_nnz - 1; ++i) {
            for (int j = 0; j < row_nnz - i - 1; ++j) {
                int idx1 = row_start + j;
                int idx2 = row_start + j + 1;

                if (C_col_ids[idx1] > C_col_ids[idx2]) {
                    // Swap column indices
                    unsigned int tmp_col = C_col_ids[idx1];
                    C_col_ids[idx1] = C_col_ids[idx2];
                    C_col_ids[idx2] = tmp_col;

                    // Swap values
                    float tmp_val = C_data[idx1];
                    C_data[idx1] = C_data[idx2];
                    C_data[idx2] = tmp_val;
                }
            }
        }
    }
}

// ========================================
// Host wrapper for Gustavson's SpGEMM
// ========================================

void gpuGustavsonSpGEMM(
    const dCSR<float>& d_A,
    const dCSR<float>& d_B,
    dCSR<float>& d_C)
{
    int M = d_A.rows;
    int K = d_A.cols;
    int N = d_B.cols;

    if (d_B.rows != static_cast<size_t>(K)) {
        throw std::runtime_error("SpGEMM dimension mismatch");
    }

    // Estimate hash table size per row (with minimum 128 for larger matrices)
    int avg_nnz_A = (d_A.nnz + M - 1) / M;
    int avg_nnz_B = (d_B.nnz + K - 1) / K;
    int estimated_output_nnz = avg_nnz_A * avg_nnz_B;
    int hash_size = std::max(128, estimated_output_nnz * HASH_SCALE * 2);  // 2x safety margin

    // Cap at reasonable size to avoid OOM
    hash_size = std::min(hash_size, 2048);

    std::cout << "Hash table size per row: " << hash_size << std::endl;

    // Allocate workspace
    unsigned int* d_hash_workspace;
    unsigned int* d_C_row_nnz;
    cudaMalloc(&d_hash_workspace, M * hash_size * sizeof(unsigned int));
    cudaMalloc(&d_C_row_nnz, M * sizeof(unsigned int));

    // Phase 1: Symbolic
    int threadsPerBlock = 128;
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;

    symbolic_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_A.row_offsets, d_A.col_ids,
        d_B.row_offsets, d_B.col_ids,
        d_C_row_nnz, d_hash_workspace,
        M, hash_size
    );
    cudaDeviceSynchronize();

    // Compute row_offsets via prefix sum
    unsigned int* d_C_row_offsets;
    cudaMalloc(&d_C_row_offsets, (M + 1) * sizeof(unsigned int));
    cudaMemset(d_C_row_offsets, 0, sizeof(unsigned int));  // First element = 0

    thrust::device_ptr<unsigned int> dev_ptr_nnz(d_C_row_nnz);
    thrust::device_ptr<unsigned int> dev_ptr_offsets(d_C_row_offsets + 1);
    thrust::inclusive_scan(dev_ptr_nnz, dev_ptr_nnz + M, dev_ptr_offsets);

    // Get total NNZ
    unsigned int nnz_C;
    cudaMemcpy(&nnz_C, d_C_row_offsets + M, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    std::cout << "Output NNZ: " << nnz_C << std::endl;

    // Allocate output
    d_C.rows = M;
    d_C.cols = N;
    d_C.nnz = nnz_C;
    cudaMalloc(&d_C.data, nnz_C * sizeof(float));
    cudaMalloc(&d_C.col_ids, nnz_C * sizeof(unsigned int));
    d_C.row_offsets = d_C_row_offsets;

    // Allocate hash table for numeric phase
    unsigned int* d_hash_keys;
    float* d_hash_vals;
    cudaMalloc(&d_hash_keys, M * hash_size * sizeof(unsigned int));
    cudaMalloc(&d_hash_vals, M * hash_size * sizeof(float));

    // Phase 2: Numeric
    numeric_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_A.data, d_A.row_offsets, d_A.col_ids,
        d_B.data, d_B.row_offsets, d_B.col_ids,
        d_C.row_offsets, d_C.col_ids, d_C.data,
        d_hash_keys, d_hash_vals,
        M, hash_size
    );
    cudaDeviceSynchronize();

    // Sort columns within each row
    sort_rows_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_C.row_offsets, d_C.col_ids, d_C.data, M
    );
    cudaDeviceSynchronize();

    // Cleanup workspace
    cudaFree(d_hash_workspace);
    cudaFree(d_C_row_nnz);
    cudaFree(d_hash_keys);
    cudaFree(d_hash_vals);
}

// ========================================
// CPU Reference (from Gustavson.cpp)
// ========================================

void cpuGustavsonSpGEMM(
    const CSR<float>& A,
    const CSR<float>& B,
    CSR<float>& C)
{
    int M = A.rows;
    int N = B.cols;

    std::vector<std::unordered_map<int, float>> rowResults(M);

    // Compute C = A × B
    for (size_t i = 0; i < A.rows; ++i) {
        int row_start = A.row_offsets[i];
        int row_end = A.row_offsets[i + 1];

        for (int idx_A = row_start; idx_A < row_end; ++idx_A) {
            int col_A = A.col_ids[idx_A];
            float val_A = A.data[idx_A];

            int row_start_B = B.row_offsets[col_A];
            int row_end_B = B.row_offsets[col_A + 1];

            for (int idx_B = row_start_B; idx_B < row_end_B; ++idx_B) {
                int col_B = B.col_ids[idx_B];
                float val_B = B.data[idx_B];

                rowResults[i][col_B] += val_A * val_B;
            }
        }
    }

    // Build CSR structure
    C.alloc(M, N, 0);  // Will resize
    C.row_offsets[0] = 0;

    size_t total_nnz = 0;
    for (size_t i = 0; i < A.rows; ++i) {
        total_nnz += rowResults[i].size();
        C.row_offsets[i + 1] = total_nnz;
    }

    C.nnz = total_nnz;
    C.data = std::make_unique<float[]>(total_nnz);
    C.col_ids = std::make_unique<unsigned int[]>(total_nnz);

    size_t idx = 0;
    for (size_t i = 0; i < A.rows; ++i) {
        // Sort columns for standard CSR format
        std::vector<std::pair<int, float>> sorted_row(rowResults[i].begin(), rowResults[i].end());
        std::sort(sorted_row.begin(), sorted_row.end());

        for (const auto& [col, val] : sorted_row) {
            C.col_ids[idx] = col;
            C.data[idx] = val;
            idx++;
        }
    }
}

#endif // GPU_GUSTAVSON_SPGEMM_CU
