#ifndef GPU_UTILS_CU
#define GPU_UTILS_CU

#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <stdexcept>
#include "CSR.h"
#include "Vector.h"

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error(cudaGetErrorString(err)); \
        } \
    } while(0)

#define CUSPARSE_CHECK(call) \
    do { \
        cusparseStatus_t status = call; \
        if (status != CUSPARSE_STATUS_SUCCESS) { \
            std::cerr << "cuSPARSE Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("cuSPARSE error"); \
        } \
    } while(0)

// ========================================
// CSR Matrix Transfer Functions
// ========================================

template<typename T>
void transferCSRToDevice(const CSR<T>& h_csr, dCSR<T>& d_csr)
{
    // Set dimensions
    d_csr.rows = h_csr.rows;
    d_csr.cols = h_csr.cols;
    d_csr.nnz = h_csr.nnz;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_csr.data, h_csr.nnz * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_csr.col_ids, h_csr.nnz * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_csr.row_offsets, (h_csr.rows + 1) * sizeof(unsigned int)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_csr.data, h_csr.data.get(),
                         h_csr.nnz * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr.col_ids, h_csr.col_ids.get(),
                         h_csr.nnz * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr.row_offsets, h_csr.row_offsets.get(),
                         (h_csr.rows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

template<typename T>
void transferCSRToHost(const dCSR<T>& d_csr, CSR<T>& h_csr)
{
    // Allocate host memory
    h_csr.alloc(d_csr.rows, d_csr.cols, d_csr.nnz);

    // Copy data from device
    CUDA_CHECK(cudaMemcpy(h_csr.data.get(), d_csr.data,
                         d_csr.nnz * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_csr.col_ids.get(), d_csr.col_ids,
                         d_csr.nnz * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_csr.row_offsets.get(), d_csr.row_offsets,
                         (d_csr.rows + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

template<typename T>
void freeDeviceCSR(dCSR<T>& d_csr)
{
    if (d_csr.data) CUDA_CHECK(cudaFree(d_csr.data));
    if (d_csr.col_ids) CUDA_CHECK(cudaFree(d_csr.col_ids));
    if (d_csr.row_offsets) CUDA_CHECK(cudaFree(d_csr.row_offsets));

    d_csr.data = nullptr;
    d_csr.col_ids = nullptr;
    d_csr.row_offsets = nullptr;
    d_csr.rows = d_csr.cols = d_csr.nnz = 0;
}

// ========================================
// Dense Matrix Transfer Functions
// ========================================

template<typename T>
void transferDenseToDevice(const DenseMatrix<T>& h_dense, dDenseMatrix<T>& d_dense)
{
    d_dense.rows = h_dense.rows;
    d_dense.cols = h_dense.cols;

    size_t total_elements = h_dense.rows * h_dense.cols;
    CUDA_CHECK(cudaMalloc(&d_dense.data, total_elements * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_dense.data, h_dense.data.get(),
                         total_elements * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void transferDenseToHost(const dDenseMatrix<T>& d_dense, DenseMatrix<T>& h_dense)
{
    h_dense.alloc(d_dense.rows, d_dense.cols);

    size_t total_elements = d_dense.rows * d_dense.cols;
    CUDA_CHECK(cudaMemcpy(h_dense.data.get(), d_dense.data,
                         total_elements * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
void freeDeviceDense(dDenseMatrix<T>& d_dense)
{
    if (d_dense.data) CUDA_CHECK(cudaFree(d_dense.data));
    d_dense.data = nullptr;
    d_dense.rows = d_dense.cols = 0;
}

// ========================================
// Allocate Device CSR (empty structure)
// ========================================

template<typename T>
void allocateDeviceCSR(dCSR<T>& d_csr, size_t rows, size_t cols, size_t nnz)
{
    d_csr.rows = rows;
    d_csr.cols = cols;
    d_csr.nnz = nnz;

    CUDA_CHECK(cudaMalloc(&d_csr.data, nnz * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_csr.col_ids, nnz * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_csr.row_offsets, (rows + 1) * sizeof(unsigned int)));
}

template<typename T>
void allocateDeviceDense(dDenseMatrix<T>& d_dense, size_t rows, size_t cols)
{
    d_dense.rows = rows;
    d_dense.cols = cols;

    CUDA_CHECK(cudaMalloc(&d_dense.data, rows * cols * sizeof(T)));
}

// ========================================
// Utility: Initialize dense matrix to zero
// ========================================

template<typename T>
void zeroDenseDevice(dDenseMatrix<T>& d_dense)
{
    size_t total = d_dense.rows * d_dense.cols;
    CUDA_CHECK(cudaMemset(d_dense.data, 0, total * sizeof(T)));
}

// Explicit template instantiations
template void transferCSRToDevice<float>(const CSR<float>&, dCSR<float>&);
template void transferCSRToHost<float>(const dCSR<float>&, CSR<float>&);
template void freeDeviceCSR<float>(dCSR<float>&);
template void allocateDeviceCSR<float>(dCSR<float>&, size_t, size_t, size_t);

template void transferCSRToDevice<double>(const CSR<double>&, dCSR<double>&);
template void transferCSRToHost<double>(const dCSR<double>&, CSR<double>&);
template void freeDeviceCSR<double>(dCSR<double>&);
template void allocateDeviceCSR<double>(dCSR<double>&, size_t, size_t, size_t);

template void transferDenseToDevice<float>(const DenseMatrix<float>&, dDenseMatrix<float>&);
template void transferDenseToHost<float>(const dDenseMatrix<float>&, DenseMatrix<float>&);
template void freeDeviceDense<float>(dDenseMatrix<float>&);
template void allocateDeviceDense<float>(dDenseMatrix<float>&, size_t, size_t);
template void zeroDenseDevice<float>(dDenseMatrix<float>&);

template void transferDenseToDevice<double>(const DenseMatrix<double>&, dDenseMatrix<double>&);
template void transferDenseToHost<double>(const dDenseMatrix<double>&, DenseMatrix<double>&);
template void freeDeviceDense<double>(dDenseMatrix<double>&);
template void allocateDeviceDense<double>(dDenseMatrix<double>&, size_t, size_t);
template void zeroDenseDevice<double>(dDenseMatrix<double>&);

#endif // GPU_UTILS_CU
