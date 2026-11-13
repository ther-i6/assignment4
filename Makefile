NVCC ?= nvcc
CUDA_ARCH ?=

BUILD_DIR := build
RESULTS_DIR := results
TARGET := $(BUILD_DIR)/lab04
DEBUG_TARGET := $(BUILD_DIR)/lab04-debug
SOURCES := \
	main.cu \
	spmv_student.cu \
	common/matrix_generator.cpp \
	common/matrix_io.cpp \
	common/benchmark_utils.cpp
HEADERS := spmv_student.cuh common/csr_matrix.hpp common/benchmark_utils.hpp
ARCH_FLAG := $(if $(strip $(CUDA_ARCH)),-arch=$(CUDA_ARCH),)
COMMON_FLAGS := -std=c++17 $(ARCH_FLAG) -I. -Icommon
RELEASE_FLAGS := -O2 -lineinfo -DNDEBUG
DEBUG_FLAGS := -O0 -g -G -lineinfo

.PHONY: all info test debug-test sanitizer benchmark report-check clean

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $@

$(TARGET): $(SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(NVCC) $(COMMON_FLAGS) $(RELEASE_FLAGS) $(SOURCES) -o $@

$(DEBUG_TARGET): $(SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(NVCC) $(COMMON_FLAGS) $(DEBUG_FLAGS) $(SOURCES) -o $@

test: $(TARGET)
	./$(TARGET) test

debug-test: $(DEBUG_TARGET)
	./$(DEBUG_TARGET) test

sanitizer: $(DEBUG_TARGET)
	@command -v compute-sanitizer >/dev/null 2>&1 || { \
		echo "未找到 compute-sanitizer，请安装对应 CUDA Toolkit 后重试。"; \
		exit 2; \
	}
	compute-sanitizer --tool memcheck --leak-check full ./$(DEBUG_TARGET) test

benchmark: $(TARGET)
	mkdir -p $(RESULTS_DIR)
	./$(TARGET) benchmark $(RESULTS_DIR)/latest.csv

report-check:
	@if grep -Eq '（请填写）|（请粘贴）|（请简述）' README.md; then \
		echo "README.md 仍有未填写的报告占位符。"; \
		exit 1; \
	else \
		echo "README.md 的固定占位符均已替换，请再检查数据和分析是否完整。"; \
	fi

info:
	@echo "CUDA 编译器："
	@$(NVCC) --version
	@echo
	@echo "GPU："
	@nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
	@echo
	@echo "CUDA_ARCH=$(if $(strip $(CUDA_ARCH)),$(CUDA_ARCH),由 nvcc 选择默认目标)"

clean:
	$(RM) -r -- $(BUILD_DIR) $(RESULTS_DIR)
