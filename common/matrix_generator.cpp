#include "csr_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace spmv_course {
namespace {

float generated_value(Index row, Index slot) {
    const std::uint32_t mixed =
        static_cast<std::uint32_t>(row) * 2654435761U +
        static_cast<std::uint32_t>(slot) * 2246822519U;
    const float magnitude = 0.25F + static_cast<float>(mixed % 1000U) / 1000.0F;
    return (mixed & 1U) == 0U ? magnitude : -magnitude;
}

void check_generator_arguments(Index rows, Index cols, Index nnz_per_row) {
    if (rows < 0 || cols < 0 || nnz_per_row < 0) {
        throw std::invalid_argument(
            "matrix dimensions and nnz_per_row must be non-negative");
    }
    if (nnz_per_row > cols) {
        throw std::invalid_argument(
            "nnz_per_row cannot exceed the number of columns");
    }
    if (rows > 0 && cols == 0 && nnz_per_row != 0) {
        throw std::invalid_argument(
            "a zero-column matrix cannot contain nonzeros");
    }
}

Index positive_mod(std::int64_t value, Index modulus) {
    const auto result = value % modulus;
    return static_cast<Index>(result < 0 ? result + modulus : result);
}

}  // namespace

void validate_csr(const CsrMatrix& matrix) {
    if (matrix.rows < 0 || matrix.cols < 0 || matrix.nnz < 0) {
        throw std::invalid_argument("CSR dimensions and nnz must be non-negative");
    }
    if (matrix.row_ptr.size() != static_cast<std::size_t>(matrix.rows) + 1U) {
        throw std::invalid_argument("row_ptr size must equal rows + 1");
    }
    if (matrix.col_idx.size() != static_cast<std::size_t>(matrix.nnz) ||
        matrix.values.size() != static_cast<std::size_t>(matrix.nnz)) {
        throw std::invalid_argument("col_idx and values sizes must equal nnz");
    }
    if (matrix.row_ptr.empty() || matrix.row_ptr.front() != 0 ||
        matrix.row_ptr.back() != matrix.nnz) {
        throw std::invalid_argument("row_ptr must start at zero and end at nnz");
    }
    for (Index row = 0; row < matrix.rows; ++row) {
        if (matrix.row_ptr[static_cast<std::size_t>(row)] >
            matrix.row_ptr[static_cast<std::size_t>(row) + 1U]) {
            throw std::invalid_argument("row_ptr must be non-decreasing");
        }
    }
    for (Index col : matrix.col_idx) {
        if (col < 0 || col >= matrix.cols) {
            throw std::invalid_argument("column index is outside the matrix");
        }
    }
}

CsrMatrix make_csr_from_dense(
    Index rows,
    Index cols,
    const std::vector<float>& dense) {
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument("matrix dimensions must be non-negative");
    }
    const auto expected_size =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    if (dense.size() != expected_size) {
        throw std::invalid_argument(
            "dense input size does not match matrix dimensions");
    }

    CsrMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.row_ptr.reserve(static_cast<std::size_t>(rows) + 1U);
    matrix.row_ptr.push_back(0);
    for (Index row = 0; row < rows; ++row) {
        for (Index col = 0; col < cols; ++col) {
            const float value = dense[
                static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) +
                static_cast<std::size_t>(col)];
            if (value != 0.0F) {
                matrix.col_idx.push_back(col);
                matrix.values.push_back(value);
            }
        }
        matrix.row_ptr.push_back(static_cast<Index>(matrix.values.size()));
    }
    matrix.nnz = static_cast<Index>(matrix.values.size());
    validate_csr(matrix);
    return matrix;
}

CsrMatrix make_banded_matrix(Index rows, Index cols, Index nnz_per_row) {
    check_generator_arguments(rows, cols, nnz_per_row);

    CsrMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.row_ptr.reserve(static_cast<std::size_t>(rows) + 1U);
    matrix.col_idx.reserve(
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(nnz_per_row));
    matrix.values.reserve(matrix.col_idx.capacity());
    matrix.row_ptr.push_back(0);

    std::vector<Index> columns(static_cast<std::size_t>(nnz_per_row));
    for (Index row = 0; row < rows; ++row) {
        for (Index slot = 0; slot < nnz_per_row; ++slot) {
            columns[static_cast<std::size_t>(slot)] = positive_mod(
                static_cast<std::int64_t>(row) - nnz_per_row / 2 + slot,
                cols);
        }
        std::sort(columns.begin(), columns.end());
        for (Index slot = 0; slot < nnz_per_row; ++slot) {
            matrix.col_idx.push_back(columns[static_cast<std::size_t>(slot)]);
            matrix.values.push_back(generated_value(row, slot));
        }
        matrix.row_ptr.push_back(static_cast<Index>(matrix.values.size()));
    }
    matrix.nnz = static_cast<Index>(matrix.values.size());
    validate_csr(matrix);
    return matrix;
}

CsrMatrix make_random_matrix(
    Index rows,
    Index cols,
    Index nnz_per_row,
    std::uint32_t seed) {
    check_generator_arguments(rows, cols, nnz_per_row);

    CsrMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.row_ptr.reserve(static_cast<std::size_t>(rows) + 1U);
    matrix.col_idx.reserve(
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(nnz_per_row));
    matrix.values.reserve(matrix.col_idx.capacity());
    matrix.row_ptr.push_back(0);

    if (cols == 0) {
        matrix.row_ptr.assign(static_cast<std::size_t>(rows) + 1U, 0);
        validate_csr(matrix);
        return matrix;
    }

    std::mt19937 generator(seed);
    std::uniform_int_distribution<Index> column_distribution(0, cols - 1);
    std::vector<Index> columns;
    columns.reserve(static_cast<std::size_t>(nnz_per_row));
    for (Index row = 0; row < rows; ++row) {
        columns.clear();
        while (columns.size() < static_cast<std::size_t>(nnz_per_row)) {
            const Index candidate = column_distribution(generator);
            if (std::find(columns.begin(), columns.end(), candidate) == columns.end()) {
                columns.push_back(candidate);
            }
        }
        std::sort(columns.begin(), columns.end());
        for (Index slot = 0; slot < nnz_per_row; ++slot) {
            matrix.col_idx.push_back(columns[static_cast<std::size_t>(slot)]);
            matrix.values.push_back(generated_value(row, slot));
        }
        matrix.row_ptr.push_back(static_cast<Index>(matrix.values.size()));
    }
    matrix.nnz = static_cast<Index>(matrix.values.size());
    validate_csr(matrix);
    return matrix;
}

CsrMatrix make_skewed_matrix(Index rows, Index cols, std::uint32_t seed) {
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument("matrix dimensions must be non-negative");
    }
    if (rows > 0 && cols == 0) {
        return CsrMatrix{
            rows,
            cols,
            0,
            std::vector<Index>(static_cast<std::size_t>(rows) + 1U, 0),
            {},
            {}};
    }

    CsrMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.row_ptr.reserve(static_cast<std::size_t>(rows) + 1U);
    matrix.row_ptr.push_back(0);
    std::vector<Index> columns;
    for (Index row = 0; row < rows; ++row) {
        Index row_nnz = 8;
        if (row % 1024 == 0) {
            row_nnz = std::min<Index>(4096, cols);
        } else if (row % 64 == 0) {
            row_nnz = std::min<Index>(256, cols);
        } else if (row % 17 == 0) {
            row_nnz = 0;
        } else {
            row_nnz = std::min<Index>(row_nnz, cols);
        }

        columns.resize(static_cast<std::size_t>(row_nnz));
        if (row_nnz > 0) {
            const std::uint64_t mixed =
                static_cast<std::uint64_t>(seed) * 6364136223846793005ULL +
                static_cast<std::uint64_t>(row) * 1442695040888963407ULL;
            const Index start =
                static_cast<Index>(mixed % static_cast<std::uint64_t>(cols));
            Index stride = static_cast<Index>(
                (mixed >> 32U) % static_cast<std::uint64_t>(cols));
            stride = std::max<Index>(1, stride);
            while (std::gcd(stride, cols) != 1) {
                ++stride;
                if (stride >= cols) {
                    stride = 1;
                }
            }
            for (Index slot = 0; slot < row_nnz; ++slot) {
                columns[static_cast<std::size_t>(slot)] = positive_mod(
                    static_cast<std::int64_t>(start) +
                        static_cast<std::int64_t>(slot) * stride,
                    cols);
            }
            std::sort(columns.begin(), columns.end());
        }

        for (Index slot = 0; slot < row_nnz; ++slot) {
            matrix.col_idx.push_back(columns[static_cast<std::size_t>(slot)]);
            matrix.values.push_back(std::abs(generated_value(row, slot)));
        }
        if (matrix.values.size() >
            static_cast<std::size_t>(std::numeric_limits<Index>::max())) {
            throw std::overflow_error(
                "generated matrix exceeds 32-bit CSR indexing");
        }
        matrix.row_ptr.push_back(static_cast<Index>(matrix.values.size()));
    }
    matrix.nnz = static_cast<Index>(matrix.values.size());
    validate_csr(matrix);
    return matrix;
}

std::vector<float> make_input_vector(Index size, std::uint32_t seed) {
    if (size < 0) {
        throw std::invalid_argument("vector size must be non-negative");
    }
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::vector<float> result(static_cast<std::size_t>(size));
    for (float& value : result) {
        value = distribution(generator);
    }
    return result;
}

std::vector<std::vector<float>> make_input_vectors(
    Index size,
    std::size_t count,
    std::uint32_t first_seed) {
    std::vector<std::vector<float>> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        result.push_back(make_input_vector(
            size,
            first_seed + static_cast<std::uint32_t>(index)));
    }
    return result;
}

std::size_t csr_storage_bytes(const CsrMatrix& matrix) {
    return matrix.row_ptr.size() * sizeof(Index) +
           matrix.col_idx.size() * sizeof(Index) +
           matrix.values.size() * sizeof(float);
}

}  // namespace spmv_course
