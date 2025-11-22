#include <iostream>
#include <vector>

// Function to compute NNZs in each row of a matrix in CSR format
std::vector<int> computeRowNNZ(const std::vector<int>& rowptr) {
    std::vector<int> nnz_per_row;
    for (size_t i = 0; i < rowptr.size() - 1; ++i) {
        nnz_per_row.push_back(rowptr[i + 1] - rowptr[i]);
    }
    return nnz_per_row;
}

// Function to compute NNZs in each column of a matrix in CSC format
std::vector<int> computeColNNZ(const std::vector<int>& colptr) {
    std::vector<int> nnz_per_col;
    for (size_t i = 0; i < colptr.size() - 1; ++i) {
        nnz_per_col.push_back(colptr[i + 1] - colptr[i]);
    }
    return nnz_per_col;
}

// int main() {
//     // Example input for matrix A in CSR format
//     std::vector<int> rowptr_A = {0, 3, 6, 8}; // rowptr for matrix A
//     std::vector<int> rowptr_B = {0, 2, 5, 7}; // rowptr for matrix B

//     // Example input for matrix C in CSC format
//     std::vector<int> colptr_C = {0, 4, 7, 9}; // colptr for matrix C

//     // Compute NNZs for rows in A
//     std::vector<int> nnz_rows_A = computeRowNNZ(rowptr_A);
//     std::cout << "NNZs in each row of A: ";
//     for (int nnz : nnz_rows_A) {
//         std::cout << nnz << " ";
//     }
//     std::cout << std::endl;

//     // Compute NNZs for rows in B
//     std::vector<int> nnz_rows_B = computeRowNNZ(rowptr_B);
//     std::cout << "NNZs in each row of B: ";
//     for (int nnz : nnz_rows_B) {
//         std::cout << nnz << " ";
//     }
//     std::cout << std::endl;

//     // Compute NNZs for columns in C
//     std::vector<int> nnz_cols_C = computeColNNZ(colptr_C);
//     std::cout << "NNZs in each column of C: ";
//     for (int nnz : nnz_cols_C) {
//         std::cout << nnz << " ";
//     }
//     std::cout << std::endl;

//     return 0;
// }