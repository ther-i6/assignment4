#include "common/benchmark_utils.hpp"
#include "common/csr_matrix.hpp"
#include "spmv_student.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using spmv_course::CsrMatrix;
using spmv_course::Index;

struct BenchmarkProtocol {
    int warmup_calls = 3;
    int batches = 7;
    std::size_t input_count = 8;
    double minimum_batch_seconds = 0.25;
    std::size_t maximum_iterations = 1U << 16U;
};

struct TestCase {
    std::string name;
    CsrMatrix matrix;
    std::vector<float> x;
};

struct BenchmarkResult {
    double median_seconds = 0.0;
    double gnnz_per_second = 0.0;
    std::size_t iterations_per_batch = 0;
};

bool same_matrix(const CsrMatrix& left, const CsrMatrix& right) {
    return left.rows == right.rows && left.cols == right.cols &&
           left.nnz == right.nnz && left.row_ptr == right.row_ptr &&
           left.col_idx == right.col_idx && left.values == right.values;
}

TestCase make_case(
    std::string name,
    Index rows,
    Index cols,
    std::vector<float> dense,
    std::vector<float> x) {
    return {
        std::move(name),
        spmv_course::make_csr_from_dense(rows, cols, dense),
        std::move(x)};
}

std::string data_root() {
    const char* configured = std::getenv("SPMV_DATA_ROOT");
    return configured != nullptr && configured[0] != '\0' ?
        configured : "data";
}

void throw_cuda(cudaError_t error, const std::string& context) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            context + ": " + cudaGetErrorString(error) +
            " (code " + std::to_string(static_cast<int>(error)) + ")");
    }
}

std::vector<TestCase> public_cases() {
    std::vector<TestCase> cases;
    cases.push_back(make_case(
        "普通 4x4",
        4,
        4,
        {0, 2, 0, 0, 1, 0, 3, 0, 0, 0, 4, 5, 0, 0, 0, 6},
        {1, 2, 3, 4}));
    cases.push_back(make_case(
        "含空行",
        4,
        3,
        {0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0},
        {1, 2, 3}));
    cases.push_back(make_case(
        "全零矩阵",
        3,
        5,
        std::vector<float>(15, 0.0F),
        {1, 2, 3, 4, 5}));
    cases.push_back(make_case(
        "宽矩阵 2x4",
        2,
        4,
        {1, 0, 2, 0, 0, 3, 0, 4},
        {1, 2, 3, 4}));
    cases.push_back(make_case(
        "高矩阵 4x2",
        4,
        2,
        {1, 0, 0, 2, 3, 4, 0, 5},
        {2, -1}));
    cases.push_back(make_case("0x0", 0, 0, {}, {}));
    cases.push_back(make_case("1x1", 1, 1, {7}, {3}));
    cases.push_back(make_case(
        "单一长行",
        1,
        4096,
        std::vector<float>(4096, 1.0F),
        std::vector<float>(4096, 0.5F)));

    CsrMatrix random = spmv_course::make_random_matrix(257, 263, 9, 2026U);
    cases.push_back({
        "固定种子随机矩阵",
        std::move(random),
        spmv_course::make_input_vector(263, 2027U)});
    CsrMatrix skewed = spmv_course::make_skewed_matrix(4096, 4096, 2028U);
    cases.push_back({
        "行长度严重不均匀",
        std::move(skewed),
        spmv_course::make_input_vector(4096, 2029U)});
    return cases;
}

bool run_case(const TestCase& test_case, bool verbose) {
    const CsrMatrix original_matrix = test_case.matrix;
    const std::vector<float> original_x = test_case.x;
    std::vector<float> expected(static_cast<std::size_t>(test_case.matrix.rows));
    spmv_course::spmv_reference(
        test_case.matrix,
        test_case.x.data(),
        expected.data());

    constexpr float guard = 1234567.0F;
    std::vector<float> guarded_output(
        static_cast<std::size_t>(test_case.matrix.rows) + 2U,
        std::numeric_limits<float>::quiet_NaN());
    guarded_output.front() = guard;
    guarded_output.back() = guard;
    float* const output = guarded_output.data() + 1;
    const cudaError_t error = spmv_student(
        test_case.matrix,
        test_case.x.data(),
        output);
    if (error != cudaSuccess) {
        if (verbose) {
            std::cerr << "[失败] " << test_case.name << "："
                      << cudaGetErrorString(error) << '\n';
        }
        return false;
    }

    const std::vector<float> actual(
        output,
        output + static_cast<std::size_t>(test_case.matrix.rows));
    const auto verification = spmv_course::verify_result(expected, actual);
    const bool guards_ok =
        guarded_output.front() == guard && guarded_output.back() == guard;
    const bool inputs_ok =
        same_matrix(test_case.matrix, original_matrix) && test_case.x == original_x;
    const bool ok = verification.ok && guards_ok && inputs_ok;
    if (verbose && !ok) {
        std::cerr << "[失败] " << test_case.name;
        if (!guards_ok) {
            std::cerr << "：检测到 y 越界写入\n";
        } else if (!inputs_ok) {
            std::cerr << "：输入矩阵或 x 被修改\n";
        } else {
            std::cerr << "：y[" << verification.first_bad_index
                      << "] 期望 " << verification.expected_at_first_bad
                      << "，实际 " << verification.actual_at_first_bad << '\n';
        }
    }
    return ok;
}

bool repeated_input_test(bool verbose) {
    const CsrMatrix matrix = spmv_course::make_banded_matrix(1024, 1031, 7);
    const auto inputs = spmv_course::make_input_vectors(matrix.cols, 8, 3000U);
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        if (!run_case({
                "同一矩阵更换 x #" + std::to_string(index + 1U),
                matrix,
                inputs[index]},
                verbose)) {
            return false;
        }
    }
    return true;
}

bool run_public_tests(bool verbose) {
    if (!spmv_student_ready()) {
        std::cerr << "[未完成] 请完成 spmv_student.cu，并让 "
                     "spmv_student_ready() 返回 true。\n";
        return false;
    }

    const auto cases = public_cases();
    int passed = 0;
    for (const auto& test_case : cases) {
        passed += run_case(test_case, verbose) ? 1 : 0;
    }
    const bool repeated_ok = repeated_input_test(verbose);
    passed += repeated_ok ? 1 : 0;
    const int total = static_cast<int>(cases.size()) + 1;
    if (passed != total) {
        std::cerr << "公开正确性测试：" << passed << '/' << total
                  << " 通过。\n";
        return false;
    }
    if (verbose) {
        std::cout << "[通过] 公开正确性测试：" << passed << '/'
                  << total << " 通过。\n";
    }
    return true;
}

double measure_calls(
    const CsrMatrix& matrix,
    const std::vector<std::vector<float>>& inputs,
    std::vector<float>& output,
    std::size_t iterations,
    std::size_t& input_index) {
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        const auto& input = inputs[input_index % inputs.size()];
        throw_cuda(
            spmv_student(matrix, input.data(), output.data()),
            "spmv_student failed during timing");
        ++input_index;
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

void verify_last_timed_result(
    const CsrMatrix& matrix,
    const std::vector<std::vector<float>>& inputs,
    const std::vector<float>& output,
    std::size_t input_index) {
    const auto& last_input = inputs[(input_index - 1U) % inputs.size()];
    std::vector<float> expected(static_cast<std::size_t>(matrix.rows));
    spmv_course::spmv_reference(matrix, last_input.data(), expected.data());
    const auto verification = spmv_course::verify_result(expected, output);
    if (!verification.ok) {
        throw std::runtime_error(
            "timed result mismatch at y[" +
            std::to_string(verification.first_bad_index) + "]");
    }
}

BenchmarkResult benchmark_matrix(
    const CsrMatrix& matrix,
    const BenchmarkProtocol& protocol) {
    auto inputs = spmv_course::make_input_vectors(
        matrix.cols,
        protocol.input_count,
        4000U);
    for (auto& input : inputs) {
        for (float& value : input) {
            value = 0.25F + std::abs(value);
        }
    }
    std::vector<float> output(static_cast<std::size_t>(matrix.rows));
    std::size_t input_index = 0;

    for (int call = 0; call < protocol.warmup_calls; ++call) {
        const auto& input = inputs[input_index % inputs.size()];
        throw_cuda(
            spmv_student(matrix, input.data(), output.data()),
            "spmv_student failed during warmup");
        ++input_index;
    }
    verify_last_timed_result(matrix, inputs, output, input_index);

    const double calibration_seconds = std::max(
        measure_calls(matrix, inputs, output, 1, input_index),
        1.0e-9);
    verify_last_timed_result(matrix, inputs, output, input_index);
    const auto requested_iterations = static_cast<std::size_t>(std::ceil(
        protocol.minimum_batch_seconds / calibration_seconds));
    const std::size_t iterations = std::clamp<std::size_t>(
        requested_iterations,
        1U,
        protocol.maximum_iterations);

    std::vector<double> seconds_per_call;
    seconds_per_call.reserve(static_cast<std::size_t>(protocol.batches));
    for (int batch = 0; batch < protocol.batches; ++batch) {
        const double elapsed = measure_calls(
            matrix,
            inputs,
            output,
            iterations,
            input_index);
        seconds_per_call.push_back(elapsed / static_cast<double>(iterations));
        verify_last_timed_result(matrix, inputs, output, input_index);
    }

    BenchmarkResult result;
    result.median_seconds = spmv_course::median(seconds_per_call);
    result.iterations_per_batch = iterations;
    result.gnnz_per_second = result.median_seconds > 0.0 ?
        static_cast<double>(matrix.nnz) / result.median_seconds / 1.0e9 : 0.0;
    return result;
}

struct CsvWriter {
    explicit CsvWriter(const std::string& path) {
        const std::filesystem::path output_path(path);
        if (output_path.has_parent_path()) {
            std::filesystem::create_directories(output_path.parent_path());
        }
        file.open(path);
        if (!file) {
            throw std::runtime_error("无法创建结果文件：" + path);
        }
        file << "matrix,rows,cols,nnz,end_to_end_ms,gnnz_per_second,"
                "iterations_per_batch\n";
    }
    std::ofstream file;
};

void benchmark_case(
    const std::string& name,
    const CsrMatrix& matrix,
    const BenchmarkProtocol& protocol,
    CsvWriter& csv) {
    const BenchmarkResult result = benchmark_matrix(matrix, protocol);
    csv.file << std::fixed << std::setprecision(8)
             << name << ',' << matrix.rows << ',' << matrix.cols << ','
             << matrix.nnz << ',' << result.median_seconds * 1000.0 << ','
             << result.gnnz_per_second << ','
             << result.iterations_per_batch << '\n';
    csv.file.flush();
    std::cout << std::left << std::setw(18) << name
              << " end-to-end=" << std::right << std::setw(10)
              << std::fixed << std::setprecision(4)
              << result.median_seconds * 1000.0 << " ms"
              << "  GNNZ/s=" << std::setprecision(6)
              << result.gnnz_per_second << '\n';
}

int run_benchmark(const std::string& output_path) {
    if (!run_public_tests(false)) {
        std::cerr << "公开检查未全部通过，拒绝开始性能测试。\n";
        return 1;
    }

    const BenchmarkProtocol protocol;
    CsvWriter csv(output_path);
    constexpr Index rows = 131072;
    benchmark_case(
        "banded_public",
        spmv_course::make_banded_matrix(rows, rows, 16),
        protocol,
        csv);
    benchmark_case(
        "random_public",
        spmv_course::make_random_matrix(rows, rows, 16, 2026U),
        protocol,
        csv);
    benchmark_case(
        "skewed_public",
        spmv_course::make_skewed_matrix(rows, rows, 2027U),
        protocol,
        csv);
    benchmark_case(
        "cage11",
        spmv_course::load_matrix_market(data_root() + "/cage11.mtx"),
        protocol,
        csv);
    std::cout << "结果已写入 " << output_path << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string command = argc > 1 ? argv[1] : "test";
        if (command == "test") {
            return run_public_tests(true) ? 0 : 1;
        }
        if (command == "benchmark") {
            const std::string output = argc > 2 ?
                argv[2] : "results/latest.csv";
            return run_benchmark(output);
        }
        std::cerr << "用法：" << argv[0]
                  << " [test | benchmark [csv路径]]\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "错误：" << error.what() << '\n';
        return 2;
    }
}
