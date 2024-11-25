#pragma once

#include "sparse_matrix.hpp"
#include "../execution/thread_pool.hpp"
#include "../execution/simd_utils.hpp"
#include <numeric>
#include <span>

namespace sparse_linalg {

namespace detail {
    template<typename Size>
    auto partition_range(Size begin, Size end, Size num_parts) {
        std::vector<Size> partitions;
        partitions.reserve(num_parts + 1);
        
        Size chunk = (end - begin) / num_parts;
        Size remainder = (end - begin) % num_parts;
        
        Size current = begin;
        partitions.push_back(current);
        
        for (Size i = 0; i < num_parts; ++i) {
            current += chunk + (i < remainder ? 1 : 0);
            partitions.push_back(current);
        }
        
        return partitions;
    }
}

template<typename T>
    requires MatrixValue<T>
class MatrixOps {
public:
    // Sequential matrix-vector multiplication
    static std::vector<T> multiply(
        const SparseMatrix<T>& matrix,
        std::span<const T> vec
    ) {
        validate_dimensions(matrix, vec);
        std::vector<T> result(matrix.rows(), T{});
        
        for (std::size_t i = 0; i < matrix.rows(); ++i) {
            auto row_vals = matrix.row_values(i);
            auto row_cols = matrix.row_indices(i);
            result[i] = sparse_dot_product(row_vals, row_cols, vec);
        }
        
        return result;
    }
    
    // Parallel and SIMD-accelerated matrix-vector multiplication
    static std::vector<T> multiply_parallel(
        const SparseMatrix<T>& matrix,
        std::span<const T> vec,
        execution::ThreadPool& pool
    ) {
        validate_dimensions(matrix, vec);
        std::vector<T> result(matrix.rows(), T{});
        
        const std::size_t num_threads = pool.thread_count();
        auto partitions = detail::partition_range(
            std::size_t{0}, matrix.rows(), num_threads
        );
        
        std::vector<std::future<void>> futures;
        futures.reserve(num_threads);
        
        for (std::size_t i = 0; i < num_threads; ++i) {
            futures.push_back(pool.submit([&, start = partitions[i], end = partitions[i + 1]]() {
                for (std::size_t row = start; row < end; ++row) {
                    auto row_vals = matrix.row_values(row);
                    auto row_cols = matrix.row_indices(row);
                    result[row] = sparse_dot_product(row_vals, row_cols, vec);
                }
            }));
        }
        
        for (auto& future : futures) {
            future.wait();
        }
        
        return result;
    }

private:
    static void validate_dimensions(const SparseMatrix<T>& matrix, std::span<const T> vec) {
        if (matrix.cols() != vec.size()) {
            throw std::invalid_argument("Vector size must match matrix columns");
        }
    }
    
    static T sparse_dot_product(
        std::span<const T> values,
        std::span<const std::size_t> indices,
        std::span<const T> vec
    ) {
        if constexpr (execution::SimdTraits<T>::is_vectorizable) {
            const std::size_t vec_size = execution::SimdTraits<T>::vector_size;
            const std::size_t vec_count = values.size() / vec_size;
            const std::size_t remainder = values.size() % vec_size;
            
            using VecType = typename execution::SimdTraits<T>::vector_type;
            auto sum = execution::SimdTraits<T>::set_zero();
            
            // Process vector-sized chunks
            for (std::size_t i = 0; i < vec_count; ++i) {
                const std::size_t base = i * vec_size;
                VecType vec_vals = execution::SimdTraits<T>::load(&values[base]);
                
                // Gather vector elements from sparse indices
                T gathered_data[vec_size];
                for (std::size_t j = 0; j < vec_size; ++j) {
                    gathered_data[j] = vec[indices[base + j]];
                }
                VecType vec_vec = execution::SimdTraits<T>::load(gathered_data);
                
                sum = execution::SimdTraits<T>::add(
                    sum,
                    execution::SimdTraits<T>::multiply(vec_vals, vec_vec)
                );
            }
            
            // Process remainder
            T result = execution::SimdTraits<T>::reduce_sum(sum);
            for (std::size_t i = vec_count * vec_size; i < values.size(); ++i) {
                result += values[i] * vec[indices[i]];
            }
            
            return result;
        } else {
            T sum{};
            for (std::size_t i = 0; i < values.size(); ++i) {
                sum += values[i] * vec[indices[i]];
            }
            return sum;
        }
    }
};

} // namespace sparse_linalg