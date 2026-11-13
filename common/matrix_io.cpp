#include "csr_matrix.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace spmv_course {
namespace {

struct Triplet {
    Index row;
    Index col;
    double value;
};

std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool read_data_line(std::istream& input, std::string& line) {
    while (std::getline(input, line)) {
        if (!line.empty() && line.front() != '%') {
            return true;
        }
    }
    return false;
}

}  // namespace

CsrMatrix load_matrix_market(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open Matrix Market file: " + path);
    }

    std::string banner;
    std::string object;
    std::string format;
    std::string field;
    std::string symmetry;
    if (!(input >> banner >> object >> format >> field >> symmetry)) {
        throw std::runtime_error("invalid Matrix Market header: " + path);
    }
    banner = lower_copy(banner);
    object = lower_copy(object);
    format = lower_copy(format);
    field = lower_copy(field);
    symmetry = lower_copy(symmetry);
    if (banner != "%%matrixmarket" || object != "matrix" ||
        format != "coordinate") {
        throw std::runtime_error(
            "only Matrix Market coordinate matrices are supported");
    }
    const bool pattern = field == "pattern";
    if (!pattern && field != "real" && field != "integer") {
        throw std::runtime_error(
            "only real, integer, and pattern matrices are supported");
    }
    const bool symmetric = symmetry == "symmetric" || symmetry == "hermitian";
    const bool skew_symmetric = symmetry == "skew-symmetric";
    if (!symmetric && !skew_symmetric && symmetry != "general") {
        throw std::runtime_error(
            "unsupported Matrix Market symmetry: " + symmetry);
    }

    input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::string line;
    if (!read_data_line(input, line)) {
        throw std::runtime_error("missing Matrix Market dimensions");
    }
    std::istringstream dimensions(line);
    std::int64_t rows64 = 0;
    std::int64_t cols64 = 0;
    std::int64_t entries64 = 0;
    if (!(dimensions >> rows64 >> cols64 >> entries64) ||
        rows64 < 0 || cols64 < 0 || entries64 < 0 ||
        rows64 > std::numeric_limits<Index>::max() ||
        cols64 > std::numeric_limits<Index>::max()) {
        throw std::runtime_error(
            "invalid or unsupported Matrix Market dimensions");
    }
    if ((symmetric || skew_symmetric) && rows64 != cols64) {
        throw std::runtime_error(
            "a symmetric Matrix Market matrix must be square");
    }

    std::vector<Triplet> entries;
    entries.reserve(
        static_cast<std::size_t>(entries64) *
        (symmetric || skew_symmetric ? 2U : 1U));
    for (std::int64_t count = 0; count < entries64; ++count) {
        if (!read_data_line(input, line)) {
            throw std::runtime_error(
                "Matrix Market file ended before all entries were read");
        }
        std::istringstream entry_stream(line);
        std::int64_t row64 = 0;
        std::int64_t col64 = 0;
        double value = 1.0;
        if (!(entry_stream >> row64 >> col64) ||
            (!pattern && !(entry_stream >> value))) {
            throw std::runtime_error("invalid Matrix Market entry");
        }
        --row64;
        --col64;
        if (row64 < 0 || row64 >= rows64 || col64 < 0 || col64 >= cols64) {
            throw std::runtime_error(
                "Matrix Market entry index is outside the matrix");
        }
        entries.push_back({
            static_cast<Index>(row64),
            static_cast<Index>(col64),
            value});
        if ((symmetric || skew_symmetric) && row64 != col64) {
            entries.push_back({
                static_cast<Index>(col64),
                static_cast<Index>(row64),
                skew_symmetric ? -value : value});
        }
    }

    std::sort(entries.begin(), entries.end(), [](const Triplet& left, const Triplet& right) {
        return std::tie(left.row, left.col) < std::tie(right.row, right.col);
    });

    std::vector<Triplet> merged;
    merged.reserve(entries.size());
    for (const Triplet& entry : entries) {
        if (!merged.empty() && merged.back().row == entry.row &&
            merged.back().col == entry.col) {
            merged.back().value += entry.value;
        } else {
            merged.push_back(entry);
        }
    }
    merged.erase(
        std::remove_if(merged.begin(), merged.end(), [](const Triplet& entry) {
            return entry.value == 0.0;
        }),
        merged.end());
    if (merged.size() >
        static_cast<std::size_t>(std::numeric_limits<Index>::max())) {
        throw std::overflow_error("matrix exceeds 32-bit CSR indexing");
    }

    CsrMatrix matrix;
    matrix.rows = static_cast<Index>(rows64);
    matrix.cols = static_cast<Index>(cols64);
    matrix.nnz = static_cast<Index>(merged.size());
    matrix.row_ptr.assign(static_cast<std::size_t>(matrix.rows) + 1U, 0);
    matrix.col_idx.reserve(merged.size());
    matrix.values.reserve(merged.size());
    for (const Triplet& entry : merged) {
        ++matrix.row_ptr[static_cast<std::size_t>(entry.row) + 1U];
        matrix.col_idx.push_back(entry.col);
        matrix.values.push_back(static_cast<float>(entry.value));
    }
    for (std::size_t row = 1; row < matrix.row_ptr.size(); ++row) {
        matrix.row_ptr[row] += matrix.row_ptr[row - 1U];
    }
    validate_csr(matrix);
    return matrix;
}

}  // namespace spmv_course
