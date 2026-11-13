#include "benchmark_utils.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <stdexcept>

namespace spmv_course {

void spmv_reference(const CsrMatrix& matrix, const float* x, float* y) {
    for (Index row = 0; row < matrix.rows; ++row) {
        double sum = 0.0;
        const Index begin = matrix.row_ptr[static_cast<std::size_t>(row)];
        const Index end = matrix.row_ptr[static_cast<std::size_t>(row) + 1U];
        for (Index index = begin; index < end; ++index) {
            const auto position = static_cast<std::size_t>(index);
            sum += static_cast<double>(matrix.values[position]) *
                   static_cast<double>(x[matrix.col_idx[position]]);
        }
        y[row] = static_cast<float>(sum);
    }
}

VerificationResult verify_result(
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    double absolute_tolerance,
    double relative_tolerance) {
    if (expected.size() != actual.size()) {
        throw std::invalid_argument("verification vectors have different sizes");
    }

    VerificationResult result;
    bool recorded_first_failure = false;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const double reference = static_cast<double>(expected[index]);
        const double observed = static_cast<double>(actual[index]);
        const double absolute_error = std::abs(reference - observed);
        const double relative_error =
            absolute_error / std::max(std::abs(reference), 1.0e-30);
        result.max_abs_error = std::max(result.max_abs_error, absolute_error);
        result.max_rel_error = std::max(result.max_rel_error, relative_error);

        const bool finite = std::isfinite(observed);
        const bool within_tolerance =
            absolute_error <= absolute_tolerance + relative_tolerance * std::abs(reference);
        if ((!finite || !within_tolerance) && !recorded_first_failure) {
            result.ok = false;
            result.first_bad_index = index;
            result.expected_at_first_bad = expected[index];
            result.actual_at_first_bad = actual[index];
            recorded_first_failure = true;
        }
    }
    return result;
}

double median(std::vector<double> values) {
    if (values.empty()) {
        throw std::invalid_argument("median needs at least one value");
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2U;
    if ((values.size() & 1U) != 0U) {
        return values[middle];
    }
    return 0.5 * (values[middle - 1U] + values[middle]);
}

std::string format_bytes(std::size_t bytes) {
    static const char* units[] = {"B", "KiB", "MiB", "GiB"};
    double value = static_cast<double>(bytes);
    std::size_t unit = 0;
    while (value >= 1024.0 && unit + 1U < std::size(units)) {
        value /= 1024.0;
        ++unit;
    }
    std::ostringstream output;
    output << std::fixed << std::setprecision(unit == 0 ? 0 : 2)
           << value << ' ' << units[unit];
    return output.str();
}

}  // namespace spmv_course
