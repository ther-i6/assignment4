#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace spmv_course {

using Index = std::int32_t;

struct CsrMatrix {
    Index rows = 0;
    Index cols = 0;
    Index nnz = 0;
    std::vector<Index> row_ptr;
    std::vector<Index> col_idx;
    std::vector<float> values;
};

void validate_csr(const CsrMatrix& matrix);

CsrMatrix make_csr_from_dense(
    Index rows,
    Index cols,
    const std::vector<float>& dense);

CsrMatrix make_banded_matrix(
    Index rows,
    Index cols,
    Index nnz_per_row);

CsrMatrix make_random_matrix(
    Index rows,
    Index cols,
    Index nnz_per_row,
    std::uint32_t seed);

CsrMatrix make_skewed_matrix(
    Index rows,
    Index cols,
    std::uint32_t seed);

CsrMatrix load_matrix_market(const std::string& path);

std::vector<float> make_input_vector(Index size, std::uint32_t seed);

std::vector<std::vector<float>> make_input_vectors(
    Index size,
    std::size_t count,
    std::uint32_t first_seed);

std::size_t csr_storage_bytes(const CsrMatrix& matrix);

}  // namespace spmv_course
