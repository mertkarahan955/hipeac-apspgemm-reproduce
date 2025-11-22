#ifndef COO_H
#define COO_H

#include <memory>
#include <cstddef>

// COO (Coordinate) Matrix Format
template<typename T>
struct COO
{
    size_t rows;        // Number of rows
    size_t cols;        // Number of columns
    size_t nnz;         // Number of non-zero elements

    std::unique_ptr<T[]> data;                    // Non-zero values [nnz]
    std::unique_ptr<unsigned int[]> row_ids;      // Row indices [nnz]
    std::unique_ptr<unsigned int[]> col_ids;      // Column indices [nnz]

    // Default constructor
    COO() : rows(0), cols(0), nnz(0) {}

    // Allocate memory
    void alloc(size_t r, size_t c, size_t n);

    // Destructor (automatic via unique_ptr)
    ~COO() = default;

    // Move semantics
    COO(COO&&) = default;
    COO& operator=(COO&&) = default;

    // Disable copy
    COO(const COO&) = delete;
    COO& operator=(const COO&) = delete;
};

// Load MatrixMarket (.mtx) file
template<typename T>
COO<T> loadMTX(const char* file);

#endif // COO_H
