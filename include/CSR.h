#ifndef CSR_H
#define CSR_H

#include <memory>
#include <cstddef>

// CSR (Compressed Sparse Row) Matrix Format
template<typename T>
struct CSR
{
    size_t rows;        // Number of rows
    size_t cols;        // Number of columns
    size_t nnz;         // Number of non-zero elements

    std::unique_ptr<T[]> data;                    // Non-zero values [nnz]
    std::unique_ptr<unsigned int[]> col_ids;      // Column indices [nnz]
    std::unique_ptr<unsigned int[]> row_offsets;  // Row offset pointers [rows+1]

    // Default constructor
    CSR() : rows(0), cols(0), nnz(0) {}

    // Allocate memory
    void alloc(size_t r, size_t c, size_t n);

    // Destructor (automatic via unique_ptr)
    ~CSR() = default;

    // Move semantics
    CSR(CSR&&) = default;
    CSR& operator=(CSR&&) = default;

    // Disable copy (use move or explicit copy)
    CSR(const CSR&) = delete;
    CSR& operator=(const CSR&) = delete;
};

// Device CSR structure for CUDA (raw pointers)
template<typename T>
struct dCSR
{
    size_t rows;
    size_t cols;
    size_t nnz;

    T* data;                  // Device pointer
    unsigned int* col_ids;    // Device pointer
    unsigned int* row_offsets; // Device pointer

    // Default constructor
    dCSR() : rows(0), cols(0), nnz(0), data(nullptr), col_ids(nullptr), row_offsets(nullptr) {}
};

// Load CSR from binary file
template<typename T>
CSR<T> loadCSR(const char* file);

// Store CSR to binary file
template<typename T>
void storeCSR(const char* file, const CSR<T>& matrix);

#endif // CSR_H
