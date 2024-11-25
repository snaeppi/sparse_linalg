#pragma once

#include <vector>
#include <span>
#include <concepts>
#include <algorithm>
#include <ranges>
#include <stdexcept>
#include <cstddef>

namespace sparse_linalg {

template<typename T>
concept MatrixValue = std::floating_point<T> || std::integral<T>;

template<typename T>
    requires MatrixValue<T>
class SparseMatrix {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using index_type = std::int32_t;

    struct CSRMatrix {
        std::vector<value_type> values;
        std::vector<size_type> col_indices;
        std::vector<size_type> row_ptrs;
    };

    SparseMatrix(size_type rows, size_type cols)
        : rows_(rows), cols_(cols) {
        data_.row_ptrs.resize(rows + 1, 0);
    }

    [[nodiscard]] auto rows() const noexcept -> size_type { return rows_; }
    [[nodiscard]] auto cols() const noexcept -> size_type { return cols_; }
    [[nodiscard]] auto nnz() const noexcept -> size_type { return data_.values.size(); }

    [[nodiscard]] auto operator()(size_type row, size_type col) const -> value_type {
        validate_indices(row, col);
        const auto row_start = data_.row_ptrs[row];
        const auto row_end = data_.row_ptrs[row + 1];
        
        auto it = std::lower_bound(
            data_.col_indices.begin() + static_cast<difference_type>(row_start),
            data_.col_indices.begin() + static_cast<difference_type>(row_end),
            col
        );
        
        if (it != data_.col_indices.begin() + static_cast<difference_type>(row_end) && *it == col) {
            const auto pos = static_cast<size_type>(std::distance(data_.col_indices.begin(), it));
            return data_.values[pos];
        }
        return value_type{};
    }

    void insert(size_type row, size_type col, value_type value) {
        validate_indices(row, col);
        if (value == value_type{}) return;

        const auto row_start = data_.row_ptrs[row];
        const auto row_end = data_.row_ptrs[row + 1];
        
        auto it = std::lower_bound(
            data_.col_indices.begin() + static_cast<difference_type>(row_start),
            data_.col_indices.begin() + static_cast<difference_type>(row_end),
            col
        );
        
        const auto pos = static_cast<size_type>(std::distance(data_.col_indices.begin(), it));
        
        if (it != data_.col_indices.begin() + static_cast<difference_type>(row_end) && *it == col) {
            data_.values[pos] = value;
        } else {
            data_.values.insert(data_.values.begin() + static_cast<difference_type>(pos), value);
            data_.col_indices.insert(it, col);
            
            for (size_type i = row + 1; i < rows_ + 1; ++i) {
                ++data_.row_ptrs[i];
            }
        }
    }

    [[nodiscard]] auto row_values(size_type row) const -> std::span<const value_type> {
        validate_row(row);
        return std::span<const value_type>(
            data_.values.begin() + static_cast<difference_type>(data_.row_ptrs[row]),
            data_.values.begin() + static_cast<difference_type>(data_.row_ptrs[row + 1])
        );
    }

    [[nodiscard]] auto row_indices(size_type row) const -> std::span<const size_type> {
        validate_row(row);
        return std::span<const size_type>(
            data_.col_indices.begin() + static_cast<difference_type>(data_.row_ptrs[row]),
            data_.col_indices.begin() + static_cast<difference_type>(data_.row_ptrs[row + 1])
        );
    }

    [[nodiscard]] const CSRMatrix& raw_data() const noexcept { return data_; }

private:
    size_type rows_;
    size_type cols_;
    CSRMatrix data_;

    void validate_indices(size_type row, size_type col) const {
        if (row >= rows_ || col >= cols_) {
            throw std::out_of_range("Matrix indices out of range");
        }
    }

    void validate_row(size_type row) const {
        if (row >= rows_) {
            throw std::out_of_range("Row index out of range");
        }
    }
};

} // namespace sparse_linalg