#include "spmv_student.cuh"

#include <cstddef>

namespace {

using spmv_course::Index;

// 每个线程块 256 个线程：足够填满 SM 的 warp 调度，同时让一行对应一个线程。
constexpr int kBlockSize = 256;

// 一线程一行的 CSR SpMV：线程编号即行号，顺序累加该行的非零元。
__global__ void spmv_csr_rows_kernel(
    const Index* __restrict__ d_row_ptr,
    const Index* __restrict__ d_col_idx,
    const float* __restrict__ d_values,
    const float* __restrict__ d_x,
    float* __restrict__ d_y,
    Index rows) {
    const Index row =
        static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }

    const Index begin = d_row_ptr[row];
    const Index end = d_row_ptr[row + 1];
    float sum = 0.0F;
    for (Index index = begin; index < end; ++index) {
        sum += d_values[index] * d_x[d_col_idx[index]];
    }
    // 空行（begin == end）在这一行也写 0，保证 y 被完整覆盖。
    d_y[row] = sum;
}

}  // namespace

bool spmv_student_ready() {
    return true;
}

cudaError_t spmv_student(
    const spmv_course::CsrMatrix& matrix,
    const float* x,
    float* y) {
    const Index rows = matrix.rows;
    const Index cols = matrix.cols;
    const Index nnz = matrix.nnz;

    // 0 行的矩阵没有任何输出元素，直接成功返回。
    if (rows <= 0) {
        return cudaSuccess;
    }

    cudaError_t status = cudaSuccess;
    Index* d_row_ptr = nullptr;
    Index* d_col_idx = nullptr;
    float* d_values = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;

    const std::size_t row_ptr_bytes =
        static_cast<std::size_t>(rows + 1) * sizeof(Index);
    const std::size_t nnz_bytes = static_cast<std::size_t>(nnz) * sizeof(Index);
    const std::size_t value_bytes = static_cast<std::size_t>(nnz) * sizeof(float);
    const std::size_t x_bytes = static_cast<std::size_t>(cols) * sizeof(float);
    const std::size_t y_bytes = static_cast<std::size_t>(rows) * sizeof(float);

    // row_ptr 一定存在（长度 rows + 1）。
    status = cudaMalloc(reinterpret_cast<void**>(&d_row_ptr), row_ptr_bytes);
    if (status == cudaSuccess) {
        status = cudaMemcpy(
            d_row_ptr,
            matrix.row_ptr.data(),
            row_ptr_bytes,
            cudaMemcpyHostToDevice);
    }

    // 只有确实存在非零元时才申请 col_idx 和 values，避免零字节申请。
    if (status == cudaSuccess && nnz > 0) {
        status = cudaMalloc(reinterpret_cast<void**>(&d_col_idx), nnz_bytes);
    }
    if (status == cudaSuccess && nnz > 0) {
        status = cudaMemcpy(
            d_col_idx,
            matrix.col_idx.data(),
            nnz_bytes,
            cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess && nnz > 0) {
        status = cudaMalloc(reinterpret_cast<void**>(&d_values), value_bytes);
    }
    if (status == cudaSuccess && nnz > 0) {
        status = cudaMemcpy(
            d_values,
            matrix.values.data(),
            value_bytes,
            cudaMemcpyHostToDevice);
    }

    // x 只在 cols > 0 时存在。
    if (status == cudaSuccess && cols > 0) {
        status = cudaMalloc(reinterpret_cast<void**>(&d_x), x_bytes);
    }
    if (status == cudaSuccess && cols > 0) {
        status = cudaMemcpy(d_x, x, x_bytes, cudaMemcpyHostToDevice);
    }

    // y 有 rows > 0 个元素，一定有内容要回传。
    if (status == cudaSuccess) {
        status = cudaMalloc(reinterpret_cast<void**>(&d_y), y_bytes);
    }

    if (status == cudaSuccess) {
        const unsigned int blocks = static_cast<unsigned int>(
            (rows + kBlockSize - 1) / kBlockSize);
        spmv_csr_rows_kernel<<<blocks, kBlockSize>>>(
            d_row_ptr,
            d_col_idx,
            d_values,
            d_x,
            d_y,
            rows);
        status = cudaGetLastError();
    }

    if (status == cudaSuccess) {
        status = cudaMemcpy(y, d_y, y_bytes, cudaMemcpyDeviceToHost);
    }

    // 统一清理：只释放成功申请到的指针，且不用清理错误覆盖首个错误。
    const cudaError_t free_y = d_y != nullptr ? cudaFree(d_y) : cudaSuccess;
    const cudaError_t free_x = d_x != nullptr ? cudaFree(d_x) : cudaSuccess;
    const cudaError_t free_values =
        d_values != nullptr ? cudaFree(d_values) : cudaSuccess;
    const cudaError_t free_col_idx =
        d_col_idx != nullptr ? cudaFree(d_col_idx) : cudaSuccess;
    const cudaError_t free_row_ptr =
        d_row_ptr != nullptr ? cudaFree(d_row_ptr) : cudaSuccess;

    if (status == cudaSuccess) {
        status = free_y;
    }
    if (status == cudaSuccess) {
        status = free_x;
    }
    if (status == cudaSuccess) {
        status = free_values;
    }
    if (status == cudaSuccess) {
        status = free_col_idx;
    }
    if (status == cudaSuccess) {
        status = free_row_ptr;
    }

    return status;
}
