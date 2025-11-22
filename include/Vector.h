#ifndef VECTOR_H
#define VECTOR_H

#include <memory>
#include <cstddef>

// Dense Vector
template<typename T>
struct Vector
{
    size_t size;
    std::unique_ptr<T[]> data;

    // Default constructor
    Vector() : size(0) {}

    // Constructor with size
    explicit Vector(size_t n) : size(n) {
        if (n > 0)
            data = std::make_unique<T[]>(n);
    }

    // Allocate memory
    void alloc(size_t n) {
        size = n;
        if (n > 0)
            data = std::make_unique<T[]>(n);
    }

    // Destructor (automatic via unique_ptr)
    ~Vector() = default;

    // Move semantics
    Vector(Vector&&) = default;
    Vector& operator=(Vector&&) = default;

    // Disable copy
    Vector(const Vector&) = delete;
    Vector& operator=(const Vector&) = delete;

    // Element access
    T& operator[](size_t i) { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }
};

// Dense Matrix (row-major)
template<typename T>
struct DenseMatrix
{
    size_t rows;
    size_t cols;
    std::unique_ptr<T[]> data;  // Row-major: data[i*cols + j]

    // Default constructor
    DenseMatrix() : rows(0), cols(0) {}

    // Constructor with dimensions
    DenseMatrix(size_t r, size_t c) : rows(r), cols(c) {
        if (r > 0 && c > 0)
            data = std::make_unique<T[]>(r * c);
    }

    // Allocate memory
    void alloc(size_t r, size_t c) {
        rows = r;
        cols = c;
        if (r > 0 && c > 0)
            data = std::make_unique<T[]>(r * c);
    }

    // Element access
    T& operator()(size_t i, size_t j) { return data[i * cols + j]; }
    const T& operator()(size_t i, size_t j) const { return data[i * cols + j]; }

    // Move semantics
    DenseMatrix(DenseMatrix&&) = default;
    DenseMatrix& operator=(DenseMatrix&&) = default;

    // Disable copy
    DenseMatrix(const DenseMatrix&) = delete;
    DenseMatrix& operator=(const DenseMatrix&) = delete;
};

// Device Dense Matrix for CUDA
template<typename T>
struct dDenseMatrix
{
    size_t rows;
    size_t cols;
    T* data;  // Device pointer (row-major)

    dDenseMatrix() : rows(0), cols(0), data(nullptr) {}
};

#endif // VECTOR_H
