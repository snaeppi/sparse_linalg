#include <doctest/doctest.h>
#include <sparse_linalg/core/sparse_matrix.hpp>

using namespace sparse_linalg;

TEST_SUITE("SparseMatrix") {
    TEST_CASE("construction") {
        SparseMatrix<double> matrix(100, 100);
        
        CHECK(matrix.rows() == 100);
        CHECK(matrix.cols() == 100);
        CHECK(matrix.nnz() == 0);
    }
    
    TEST_CASE("element insertion and retrieval") {
        SparseMatrix<double> matrix(10, 10);
        
        SUBCASE("single element") {
            matrix.insert(0, 0, 1.0);
            CHECK(matrix.nnz() == 1);
            CHECK(matrix(0, 0) == doctest::Approx(1.0));
            CHECK(matrix(0, 1) == doctest::Approx(0.0));
        }
        
        SUBCASE("multiple elements") {
            matrix.insert(1, 0, 2.0);
            matrix.insert(1, 2, 3.0);
            CHECK(matrix.nnz() == 2);
            CHECK(matrix(1, 0) == doctest::Approx(2.0));
            CHECK(matrix(1, 1) == doctest::Approx(0.0));
            CHECK(matrix(1, 2) == doctest::Approx(3.0));
        }
        
        SUBCASE("update existing element") {
            matrix.insert(5, 5, 1.0);
            matrix.insert(5, 5, 2.0);
            CHECK(matrix.nnz() == 1);
            CHECK(matrix(5, 5) == doctest::Approx(2.0));
        }
    }
    
    TEST_CASE("bounds checking") {
        SparseMatrix<double> matrix(5, 5);
        
        CHECK_THROWS_AS(matrix.insert(5, 0, 1.0), std::out_of_range);
        CHECK_THROWS_AS(matrix.insert(0, 5, 1.0), std::out_of_range);
        CHECK_THROWS_AS([[maybe_unused]] auto x = matrix(5, 0), std::out_of_range);
        CHECK_THROWS_AS([[maybe_unused]] auto x = matrix(0, 5), std::out_of_range);
    }
    
    TEST_CASE("row access") {
        SparseMatrix<double> matrix(5, 5);
        matrix.insert(2, 0, 1.0);
        matrix.insert(2, 2, 2.0);
        matrix.insert(2, 4, 3.0);
        
        auto values = matrix.row_values(2);
        auto indices = matrix.row_indices(2);
        
        REQUIRE(values.size() == 3);
        REQUIRE(indices.size() == 3);
        
        CHECK(values[0] == doctest::Approx(1.0));
        CHECK(values[1] == doctest::Approx(2.0));
        CHECK(values[2] == doctest::Approx(3.0));
        
        CHECK(indices[0] == 0);
        CHECK(indices[1] == 2);
        CHECK(indices[2] == 4);
    }
}