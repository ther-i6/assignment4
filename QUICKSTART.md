# 实验四快速开始

本实验只有一个任务：在 Host 端输入/输出接口不变的前提下，完成一次完整的 CUDA CSR SpMV。概念、边界和评分请看 `GUIDE.md`，实验记录填写在 `README.md`。

## 1. 检查环境

需要 NVIDIA GPU、驱动、CUDA Toolkit（含 `nvcc`），建议同时安装 `compute-sanitizer`。

遇到环境或 API 问题时，优先查阅 NVIDIA 官方资料：

- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/)
- [NVIDIA GPU Compute Capability 列表](https://developer.nvidia.com/cuda-gpus)

```bash
cd assignment4
make info
make
```

若需要为本机 GPU 指定架构，可先在 [GPU Compute Capability 列表](https://developer.nvidia.com/cuda-gpus) 查找本机型号，再传入例如：

```bash
make CUDA_ARCH=sm_89
```

不要照抄 `sm_89` 到所有机器；它是正式 NVIDIA L4 评测环境的架构。本地性能只用于观察，不用于班级排名。

## 2. 只修改一个文件

只修改 `spmv_student.cu`：

1. 让 `spmv_student_ready()` 返回 `true`；
2. 申请 CSR、`x`、`y` 所需设备内存；
3. 完成 H2D、kernel、D2H 和资源释放；
4. 返回首个 CUDA 错误，成功时返回 `cudaSuccess`。

先从“一线程处理一行”得到正确基线，再尝试课件范围内的线程映射、块大小、共享内存或多个 kernel。每次调用必须独立，不能跨调用缓存。

## 3. 检查和测量

```bash
make test
make debug-test
make sanitizer
make benchmark
```

公开基准写入 `results/latest.csv`。端到端时间包括设备内存申请、H2D、kernel、D2H 和释放；Host 端读取、生成、内存申请、CPU 参考计算与验证不计时。

如果本机安装了 NVIDIA Nsight 工具，可以用它观察时间线和某个 kernel 的指标，这部分只用于学习和填写报告。官方入口：[Nsight Systems](https://docs.nvidia.com/nsight-systems/)、[Nsight Compute](https://docs.nvidia.com/nsight-compute/)。简要命令见 `GUIDE.md` 的“性能分析工具入门”。

## 4. 填写报告并提交

将环境、设计、检查结果、公开数据和简短分析填写到 `README.md`，然后执行：

```bash
make report-check
make clean
```

提交完整仓库压缩包，不要只交 `spmv_student.cu`。不要加入 `build/`、`results/`、ZIP、缓存。

开放期间排行榜显示排名、综合 GNNZ/s、正确性、报告完成度和累计提交次数。截止后在 NVIDIA L4 上统一重测 12 个隐藏矩阵并计算 20 分性能成绩；补交没有性能分。
