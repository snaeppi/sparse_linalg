#include <sparse_linalg/core/sparse_matrix.hpp>
#include <sparse_linalg/core/matrix_ops.hpp>
#include <sparse_linalg/execution/thread_pool.hpp>
#include <iostream>
#include <iomanip>

using namespace sparse_linalg;

int main() {
    // Create a small sparse matrix
    SparseMatrix<double> matrix(5, 5);
    
    // Insert some values
    matrix.insert(0, 0, 2.0);
    matrix.insert(0, 1, -1.0);
    matrix.insert(1, 0, -1.0);
    matrix.insert(1, 1, 2.0);
    matrix.insert(1, 2, -1.0);
    matrix.insert(2, 1, -1.0);
    matrix.insert(2, 2, 2.0);
    matrix.insert(2, 3, -1.0);
    matrix.insert(3, 2, -1.0);
    matrix.insert(3, 3, 2.0);
    matrix.insert(3, 4, -1.0);
    matrix.insert(4, 3, -1.0);
    matrix.insert(4, 4, 2.0);
    
    std::vector<double> vec{1.0, 1.0, 1.0, 1.0, 1.0};
    
    // Sequential multiplication
    std::cout << "Sequential multiplication result:\n";
    auto result1 = MatrixOps<double>::multiply(matrix, vec);
    for (auto val : result1) {
        std::cout << std::setw(8) << val << " ";
    }
    std::cout << "\n\n";
    
    // Parallel multiplication (automatically uses SIMD when available)
    execution::ThreadPool pool;
    std::cout << "Parallel multiplication result:\n";
    auto result2 = MatrixOps<double>::multiply_parallel(matrix, vec, pool);
    for (auto val : result2) {
        std::cout << std::setw(8) << val << " ";
    }
    std::cout << "\n";
    
    bool results_match = true;
    for (std::size_t i = 0; i < result1.size(); ++i) {
        if (std::abs(result1[i] - result2[i]) > 1e-10) {
            results_match = false;
            break;
        }
    }
    
    std::cout << "\nResults " << (results_match ? "match" : "don't match") << "\n";
    
    std::cout << "\nMatrix information:\n";
    std::cout << "Size: " << matrix.rows() << " x " << matrix.cols() << "\n";
    std::cout << "Non-zeros: " << matrix.nnz() << "\n";
    
    // Calculate density
    const auto total_elements = static_cast<double>(matrix.rows()) * 
                              static_cast<double>(matrix.cols());
    const auto density = static_cast<double>(matrix.nnz()) / total_elements * 100.0;
    std::cout << "Density: " << density << "%\n";
    
    return 0;
}