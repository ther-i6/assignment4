# 实验四指导书：CUDA CSR SpMV

## 1. 实验目标

实验一和实验二分别完成了串行与 OpenMP CSR SpMV；实验三在 DCU 平台观察了 HIP 程序、数学库和性能分析。本实验把同一计算迁移到 NVIDIA GPU，并要求一次函数调用完成从 Host 数据到 Host 结果的全部 CUDA 生命周期。

完成后，你应当能够：

1. 用 Grid、Block、Thread 说明计算工作如何映射到 GPU；
2. 正确使用 `cudaMalloc`、`cudaMemcpy`、kernel 启动和 `cudaFree`；
3. 检查 CUDA 错误并在失败路径上释放已经申请的资源；
4. 处理空行、矩形矩阵、长行和不均匀行等 CSR 边界情况；
5. 用端到端数据比较不同线程映射和存储访问方案。

本实验只有一个最终任务，没有编号小实验。请先得到正确基线，再进行性能优化。

## 2. 数据与接口

学生只修改 `spmv_student.cu`。接口在 `spmv_student.cuh` 中：

```cpp
bool spmv_student_ready();

cudaError_t spmv_student(
    const spmv_course::CsrMatrix& matrix,
    const float* x,
    float* y);
```

进入函数时：

- `matrix.row_ptr`、`matrix.col_idx`、`matrix.values` 位于 Host；
- `x` 位于 Host，长度为 `matrix.cols`；
- `y` 位于 Host，长度为 `matrix.rows`；
- `row_ptr` 长度为 `rows + 1`，`row_ptr[rows] == nnz`。

返回前必须完成计算并把全部 `y` 写回 Host。不得修改矩阵和 `x`。初始模板让 `spmv_student_ready()` 返回 `false`，因此可以编译，但公开检查会明确报告“未完成”。

## 3. 建议的正确基线

建议先采用一线程处理一行：线程编号对应行号；该线程按 `row_ptr[row]` 到 `row_ptr[row + 1]` 顺序累加非零元并写 `y[row]`。这不是最终算法限制，只是最容易验证的起点。

一次调用至少包括：

1. 根据 `rows`、`cols`、`nnz` 计算字节数；
2. 为非空数组申请 Device 内存；
3. 复制 `row_ptr`、`col_idx`、`values` 和 `x` 到 Device；
4. 启动 kernel，并检查启动错误；
5. 等待计算完成，把 `y` 复制回 Host；
6. 释放本次调用申请的每块 Device 内存；
7. 成功返回 `cudaSuccess`，失败返回实际遇到的首个 CUDA 错误。

特别注意：CUDA 对零字节申请的行为不应成为程序正确性的前提。`rows == 0`、`cols == 0` 或 `nnz == 0` 时，只对确实需要的数组申请和复制；全零但有若干行的矩阵仍必须把每个 `y[row]` 写为零。

## 4. 错误处理与清理

程序可能在第二次或第三次 `cudaMalloc`、某次 `cudaMemcpy`、kernel 启动或结果回传时失败。CUDA API 的参数和返回值可查阅 NVIDIA 的 [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)。此时应：

- 保存遇到的首个错误；
- 进入统一清理路径；
- 只释放已经成功获得的指针；
- 不用后续清理错误覆盖首个计算错误；
- 函数内不抛出异常穿过评测边界。

`make debug-test` 使用 `-G` 生成调试版本；`make sanitizer` 使用 Compute Sanitizer 检查越界和设备内存泄漏。仅仅数值正确不能证明没有越界。

## 5. 可以探索的优化

在课件范围内，可以比较：

- 线程块大小和 grid 大小；
- 一线程处理一行、多个线程协作处理一行，或按行结构选择不同映射；
- grid-stride 循环；
- 一个或多个 kernel；
- 寄存器局部累加；
- 全局内存访问是否便于合并；
- 共享内存或动态共享内存；
- `__syncthreads()` 和原子操作；
- `cudaMemcpyAsync` 与默认流上的同步；
- 查询设备属性后选择普通参数。

每种机制都应服务于具体瓶颈。共享内存并不会自动加速，额外同步、原子竞争、空闲线程和更多 kernel 启动都可能使端到端时间变长。公开矩阵只用于观察，最终方案要面对未知的行长度和列访问分布。

## 6. 优化边界

允许使用课程课件中已经讲过的 CUDA Runtime、kernel、Grid/Block/Thread、存储层次、同步、原子操作和普通 C++ 循环优化。

禁止：

- cuSPARSE、Thrust、CUB、Eigen 或其他外部计算库；
- 用户创建的多流流水线、CUDA Graph、Cooperative Groups、动态并行；
- warp shuffle 等课件未讲授的专用 warp 内建、Tensor Core、WMMA、内联 PTX；
- 统一内存、零拷贝、纹理内存或表面内存；
- pthread、`std::thread`、OpenMP 或多 GPU；
- 把 CSR 做矩阵格式转换或重排矩阵；
- 根据公开矩阵名称、尺寸或数据硬编码矩阵特例；
- 使用全局或静态对象进行跨调用缓存；
- 在学生函数内读取计时器、环境变量、文件或评测进程信息。

可以在一次调用内部申请临时 Device 缓冲区；函数返回前必须释放。评测只提取 `spmv_student.cu`，与可信公共组件重新编译，因此修改其他文件不会改变正式测试。

## 7. 正确性检查

公开测试覆盖基础方阵、空行、全零矩阵、宽/高矩形矩阵、0×0、单元素、长行、随机矩阵、严重不均匀行、重复调用和 8 组不同 `x`。同时检查：

- CUDA 返回状态；
- 数值误差和 NaN/Inf；
- `y` 是否完整覆盖及是否越界；
- 矩阵与 `x` 是否被修改；
- 不同矩阵连续调用时是否残留旧状态。

命令：

```bash
make test
make debug-test
make sanitizer
```

隐藏测试还会改变矩阵规模、行数是否为块大小整数倍、空行位置、长行位置和数值。不要用某一公开输出代替边界推理。

## 8. 性能分析工具入门

这一步是辅助观察，不改变正式计时规则，也不要求把分析报告文件提交。先确认工具是否存在：

```bash
command -v nsys
command -v ncu
```

如果命令不存在，先看 NVIDIA 的 [Nsight Systems 文档](https://docs.nvidia.com/nsight-systems/) 和 [Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/)，或直接跳过本节，不影响 `make test` 和 `make benchmark`。

### 8.1 用 Nsight Systems 看端到端时间线

把分析文件放在 `/tmp`，避免污染提交目录：

```bash
mkdir -p /tmp/assignment4-profile
nsys profile --stats=true --force-overwrite=true \
  --output=/tmp/assignment4-profile/spmv-nsys \
  ./build/lab04 benchmark /tmp/assignment4-profile/nsys.csv
```

命令结束后，终端会给出统计摘要；也可以用 Nsight Systems 图形界面打开 `/tmp/assignment4-profile/spmv-nsys.nsys-rep`。先观察三类时间是否占主导：Host 到 Device/Device 到 Host 的拷贝、kernel 执行、以及两者之间的空隙。报告只需记录一个具体观察，例如“端到端时间中 H2D/D2H 占比较高”。

### 8.2 用 Nsight Compute 看一个 kernel

```bash
ncu --set basic --target-processes all \
  --export=/tmp/assignment4-profile/spmv-ncu \
  ./build/lab04 benchmark /tmp/assignment4-profile/ncu.csv
ncu --import /tmp/assignment4-profile/spmv-ncu.ncu-rep
```

在报告中任选一个 kernel，记录 `Duration`，并再记录 `Memory Throughput`、`Compute Throughput` 或 `Achieved Occupancy` 中的一个值。kernel 名称会随你的实现变化，不要照抄示例名称；在 `ncu` 输出中选择实际的 SpMV kernel 即可。Nsight Compute 可能显著降低运行速度，所以这里的数值不能与正式排行榜直接比较。

完成后可删除临时文件：

```bash
rm -rf /tmp/assignment4-profile
```

## 9. 端到端计时规则

正式计时只包围一次完整的 `spmv_student` 调用，包含：

- Device 内存申请；
- CSR 与 `x` 的 H2D；
- 一个或多个 kernel 及必要同步；
- `y` 的 D2H；
- Device 内存释放。

矩阵读取/生成、Host 内存申请、CPU 参考计算和结果验证排除。预热 3 次完整调用；自动增加迭代次数使每批不少于 250 ms；测量 7 批并取单次时间中位数；循环使用 8 组 `x`。

公开基准：

```bash
make benchmark
```

输出 `banded_public`、`random_public`、`skewed_public` 和 `cage11` 的端到端中位时间与 GNNZ/s。本地 GPU、驱动、功耗和后台负载不同，因此本地性能只用于观察优化趋势，不能与其他同学的不同机器直接比较。

## 10. 评分与排行榜

总分 100 分：

- 正确性与工程质量 70 分；
- 简短实验报告 10 分；
- 截止后性能 20 分。

正确性 70 分包括：构建、接口及确实使用 CUDA 10 分；基础/空行/矩形 CSR 15 分；随机/真实/长行/严重不均匀矩阵 20 分；重复调用/更换 `x`/输出完整覆盖 15 分；CUDA 错误、越界、输入不变和数值合法性 10 分。

正式性能使用 NVIDIA L4。12 个隐藏矩阵包括 6 个合成矩阵和 6 个真实矩阵，覆盖规则行、随机列访问、空行、长行、重尾分布和严重不均匀结构。每个矩阵吞吐率为 $g_{i,d}=\mathrm{nnz}_d/T_{i,d}$，综合指标为

$$
G_i=\exp\left(\frac{1}{12}\sum_d\ln g_{i,d}\right).
$$

开放期间显示排名、$G_i$、正确性、报告完成度和累计提交次数；每次启动评测都计入提交次数，不论是否有效，不显示暂定性能分。截止后在 L4 上统一串行复测，按

$$
P_i=20\frac{G_i}{G_{\max}^{student}}
$$

计算性能分。前三名进行规则检查和三轮独立复测；补交没有性能分。

## 11. 完成标准

提交前至少确认：

```bash
make clean
make
make test
make debug-test
make sanitizer
make benchmark
make report-check
```

最终提交完整仓库压缩包。保留 `assignment4/README.md`、`GUIDE.md`、`QUICKSTART.md`、`Makefile`、`common/`、`data/`、`main.cu`、`spmv_student.cuh` 和你的 `spmv_student.cu`；不提交 `build/`、`results/`、ZIP。
