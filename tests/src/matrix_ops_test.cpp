#include <doctest/doctest.h>
#include <sparse_linalg/core/sparse_matrix.hpp>
#include <sparse_linalg/core/matrix_ops.hpp>
#include <sparse_linalg/execution/thread_pool.hpp>

using namespace sparse_linalg;

TEST_SUITE("MatrixOperations") {
    TEST_CASE("matrix-vector multiplication") {
        SparseMatrix<double> matrix(3, 3);
        matrix.insert(0, 0, 1.0);
        matrix.insert(0, 1, 2.0);
        matrix.insert(1, 1, 3.0);
        matrix.insert(2, 1, 4.0);
        matrix.insert(2, 2, 5.0);
        
        std::vector<double> vec{1.0, 2.0, 3.0};
        
        SUBCASE("sequential multiplication") {
            auto result = MatrixOps<double>::multiply(matrix, vec);
            REQUIRE(result.size() == 3);
            CHECK(result[0] == doctest::Approx(5.0));  // 1*1 + 2*2 + 0*3
            CHECK(result[1] == doctest::Approx(6.0));  // 0*1 + 3*2 + 0*3
            CHECK(result[2] == doctest::Approx(23.0)); // 0*1 + 4*2 + 5*3
        }
        
        SUBCASE("parallel multiplication") {
            execution::ThreadPool pool(4);
            auto result = MatrixOps<double>::multiply_parallel(matrix, vec, pool);
            REQUIRE(result.size() == 3);
            CHECK(result[0] == doctest::Approx(5.0));
            CHECK(result[1] == doctest::Approx(6.0));
            CHECK(result[2] == doctest::Approx(23.0));
        }
    }
    
    TEST_CASE("error handling") {
        SparseMatrix<double> matrix(3, 3);
        std::vector<double> vec{1.0, 2.0};  // Wrong size
        
        SUBCASE("incompatible dimensions") {
            CHECK_THROWS_AS(MatrixOps<double>::multiply(matrix, vec), 
                          std::invalid_argument);
            
            execution::ThreadPool pool(4);
            CHECK_THROWS_AS(MatrixOps<double>::multiply_parallel(matrix, vec, pool), 
                          std::invalid_argument);
        }
    }
    
    TEST_CASE("large matrix multiplication") {
        const std::size_t size = 1000;
        SparseMatrix<double> matrix(size, size);
        
        for (std::size_t i = 0; i < size; ++i) {
            matrix.insert(i, i, 2.0);
            if (i > 0) matrix.insert(i, i-1, -1.0);
            if (i < size-1) matrix.insert(i, i+1, -1.0);
        }
        
        std::vector<double> vec(size, 1.0);
        
        auto result1 = MatrixOps<double>::multiply(matrix, vec);
        
        execution::ThreadPool pool(4);
        auto result2 = MatrixOps<double>::multiply_parallel(matrix, vec, pool);
        
        REQUIRE(result1.size() == result2.size());
        for (std::size_t i = 0; i < result1.size(); ++i) {
            CHECK(result1[i] == doctest::Approx(result2[i]));
        }
    }
}