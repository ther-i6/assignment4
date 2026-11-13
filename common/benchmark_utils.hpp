#pragma once

#include "csr_matrix.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace spmv_course {

struct VerificationResult {
    bool ok = true;
    std::size_t first_bad_index = 0;
    float expected_at_first_bad = 0.0F;
    float actual_at_first_bad = 0.0F;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
};

void spmv_reference(const CsrMatrix& matrix, const float* x, float* y);

VerificationResult verify_result(
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    double absolute_tolerance = 1.0e-5,
    double relative_tolerance = 1.0e-4);

double median(std::vector<double> values);

std::string format_bytes(std::size_t bytes);

}  // namespace spmv_course
