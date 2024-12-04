#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

typedef vector<vector<int>> Matrix;

struct RowPair {
    int rowA;
    int rowB;
    int sparsity;
};

// Function to calculate the mean sparsity of two rows
int calculateSparsityMean(const vector<int>& rowA, const vector<int>& rowB) {
    return (count(rowA.begin(), rowA.end(), 0) + count(rowB.begin(), rowB.end(), 0)) / 2;
}

// Function to sort RowPair by sparsity in descending order
bool compareBySparsity(const RowPair& a, const RowPair& b) {
    return a.sparsity > b.sparsity;
}

// Adaptive Panel Allocation function
void adaptivePanelAllocation(const Matrix& A, const Matrix& B, Matrix& C, double K) {
    int numRowsA = A.size();
    int numRowsB = B.size();

    vector<RowPair> rowPairs;

    // Step 1: Preprocessing
    // Analyze NNZs and reorder
    for (int i = 0; i < numRowsA; ++i) {
        for (int j = 0; j < numRowsB; ++j) {
            RowPair rp;
            rp.rowA = i;
            rp.rowB = j;
            rp.sparsity = calculateSparsityMean(A[i], B[j]);
            rowPairs.push_back(rp);
        }
    }

    // Sort the row pairs by sparsity
    sort(rowPairs.begin(), rowPairs.end(), compareBySparsity);

    // Calculate GPU-to-CPU allocation ratio
    double R = K / (K + 1);
    int gpuThreshold = static_cast<int>(R * rowPairs.size());

    // Step 2: Panel Allocation and Computation
    // GPU allocation
    for (int G = 0; G < gpuThreshold; ++G) {
        int rowAIndex = rowPairs[G].rowA;
        int rowBIndex = rowPairs[G].rowB;

        // GPU computation (placeholder for actual GPU kernel calls)
        for (int k = 0; k < A[rowAIndex].size(); ++k) {
            C[rowAIndex][k] += A[rowAIndex][k] * B[rowBIndex][k];
        }
    }

    // CPU allocation
    for (int C_idx = gpuThreshold; C_idx < rowPairs.size(); ++C_idx) {
        int rowAIndex = rowPairs[C_idx].rowA;
        int rowBIndex = rowPairs[C_idx].rowB;

        // CPU computation
        for (int k = 0; k < A[rowAIndex].size(); ++k) {
            C[rowAIndex][k] += A[rowAIndex][k] * B[rowBIndex][k];
        }
    }

}

// int main() {
//     // Example usage
//     Matrix A = {{1, 0, 0}, {0, 2, 0}, {3, 0, 0}};
//     Matrix B = {{0, 4, 0}, {0, 0, 5}, {6, 0, 0}};
//     Matrix C(3, vector<int>(3, 0));

//     double K = 1.5; // Computational coefficient

//     adaptivePanelAllocation(A, B, C, K);

//     // Output the result matrix
//     cout << "Resultant Matrix C:" << endl;
//     for (const auto& row : C) {
//         for (int val : row) {
//             cout << val << " ";
//         }
//         cout << endl;
//     }

//     return 0;
// }