#pragma once

#include "common/csr_matrix.hpp"

#include <cuda_runtime.h>

bool spmv_student_ready();

cudaError_t spmv_student(
    const spmv_course::CsrMatrix& matrix,
    const float* x,
    float* y);
