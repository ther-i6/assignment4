# 实验四报告：CUDA CSR SpMV

本文件既是简短实验报告模板，也是最终提交的一部分。只修改报告占位符和 `spmv_student.cu`，不要改动评测框架、公共头文件或数据。

## 实验报告（10 分）

### 1. 基本信息与环境（2 分）

| 项目 | 内容 |
| --- | --- |
| 姓名 | 马浩宇（请自行确认） |
| 学号 | （请填写） |
| 班级 | （请填写） |
| 实验日期 | 2026-11-13 |
| GPU 型号 | NVIDIA L4（23034 MiB） |
| CUDA Toolkit / `nvcc` 版本 | 12.4 / V12.4.99 |
| NVIDIA 驱动版本 | 550.90.07 |
| 编译参数 | `-std=c++17 -I. -Icommon -O2 -lineinfo -DNDEBUG -arch=sm_89`（Release）与 `-O0 -g -G -lineinfo -arch=sm_89`（Debug） |

粘贴 `make info` 的关键输出：

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Feb_27_16:19:38_PST_2024
Cuda compilation tools, release 12.4, V12.4.99
Build cuda_12.4.r12.4/compiler.33961263_0

name, driver_version, memory.total [MiB]
NVIDIA L4, 550.90.07, 23034 MiB

CUDA_ARCH=? nvcc ??????
```

### 2. 实现设计（3 分）

请简述线程与矩阵工作的对应关系、线程块大小、内存申请与复制流程，以及你为不同长度稀疏行所做的优化。若使用共享内存、同步或原子操作，请说明它们保护或复用的具体数据；若未使用，也请说明原因。

本实现为一线程一行的正确基线。

线程映射：网格采用 `grid = ceil(rows / 256)`、`block = 256`。核函数中 `row = blockIdx.x * blockDim.x + threadIdx.x`，`row >= rows` 的线程立即返回；其余线程负责第 `row` 行，按 `row_ptr[row]` 到 `row_ptr[row + 1]` 顺序读取 `values` 与 `col_idx`，在寄存器中累加后写回 `y[row]`。因此每个输出元素只被一个线程写一次，不存在写冲突，不需要原子操作。

内存流程：每次调用都根据 `rows`、`cols`、`nnz` 计算字节数，用 `cudaMalloc` 依次申请 `row_ptr`（`rows + 1` 个 `Index`）、`col_idx` 与 `values`（各 `nnz` 个元素）、`x`（`cols` 个 `float`）和 `y`（`rows` 个 `float`）的设备内存，再用 4 次 `cudaMemcpy`（Host→Device）把 CSR 与 `x` 送上去；kernel 启动后用 `cudaGetLastError()` 检查启动错误，然后用 `cudaMemcpy`（Device→Host）把 `y` 拷回，最后 `cudaFree` 释放本次申请的全部指针并返回首个遇到的 CUDA 错误。所有指针都是函数局部变量，函数返回前全部释放，因此不同矩阵、不同 `x` 的连续调用之间没有残留状态。

未使用共享内存、`__syncthreads()`、原子操作或多 kernel：一线程一行时一行的工作只由一个线程完成，共享内存无数据可复用，同步与原子只会增加开销。对于长行，该线程会顺序遍历整行，负载集中在少数线程上，这也是后续可优化的方向。

### 3. 正确性与资源检查（2 分）

| 命令 | 是否通过 | 关键结果 |
| --- | --- | --- |
| `make test` | 通过 | 公开正确性测试 11/11 通过（含空行、全零矩阵、宽/高矩形、0x0、1x1、长行、固定种子随机、严重不均匀行、更换 8 组 x 的重复调用测试） |
| `make debug-test` | 通过 | 调试版 11/11 通过；`-G` 覆盖 `-lineinfo` 的 ptxas 警告属预期现象 |
| `make sanitizer` | 跳过 | 在线实验环境未安装 `compute-sanitizer`，Makefile 提示"未找到 compute-sanitizer" |
| `make report-check` | 通过 | 固定占位符已全部替换 |

说明你的实现如何处理空矩阵、全零矩阵、矩形矩阵、空行、长行，以及 CUDA 调用中途失败时已经申请资源的释放。

边界处理：
- 0×0 矩阵（`rows == 0`）——函数入口处直接返回 `cudaSuccess`，不做任何设备操作。
- 全零矩阵（`nnz == 0`）——只申请并复制 `row_ptr`；`col_idx` 与 `values` 因 `nnz == 0` 被显式跳过，避免把零字节申请行为当正确性前提；kernel 中 `begin == end` 时循环不执行，线程仍写 `y[row] = 0.0F`，保证 `y` 被完整覆盖。
- 矩形矩阵（`rows != cols`）——`col_idx` 访问 `x` 时按实际 `cols` 尺寸申请和复制，kernel 中 `x[d_col_idx[index]]` 自然取到正确位置。
- 空行——`begin == end`，循环 0 次，线程写 `y[row] = 0.0F`，输出完整覆盖。
- 单一长行——只有一个线程承担整行的累加；其余线程 `row >= rows` 早退。

错误释放：函数用局部指针 `d_row_ptr`、`d_col_idx`、`d_values`、`d_x`、`d_y`，初始为 `nullptr`。每步 `cudaMalloc` / `cudaMemcpy` 都在 `status == cudaSuccess` 时才继续，失败时 `status` 被赋值为首个错误码，后续步骤全部跳过。最后进入统一清理：对非 `nullptr` 的指针分别调用 `cudaFree`，清理阶段产生的错误只在 `status == cudaSuccess` 时才覆盖，确保返回值始终是首个遇到的计算错误。函数返回前所有本次申请的指针均被释放，不存在跨调用残留。

### 4. 公开端到端性能（2 分）

执行 `make benchmark`，填写 `results/latest.csv` 的中位数。端到端时间包含每次调用中的设备内存申请、H2D、核函数、D2H 和释放；不包含矩阵读取/生成、Host 内存申请、CPU 参考结果和验证。

| 矩阵 | rows | cols | nnz | 端到端时间/ms | GNNZ/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| banded_public | 131072 | 131072 | 2097152 | 3.862976 | 0.542885 |
| random_public | 131072 | 131072 | 2097152 | 2.628629 | 0.797812 |
| skewed_public | 131072 | 131072 | 1987280 | 2.678113 | 0.742045 |
| cage11 | 39082 | 39082 | 559722 | 1.795307 | 0.311770 |

### 5. 结果分析（1 分）

结合至少两类矩阵，说明端到端开销、访存合并、线程利用率和行长度分布如何影响结果不得只写"GPU 更快"或只粘贴输出。

**banded vs random（同规模 2M NNZ，行结构不同）**：banded 是规则带状（每行 16 个非零元），random 是固定种子随机分布。两者 NNZ 相同，banded 反而比 random 慢 47%（3.86 ms vs 2.63 ms），这在一线程一行实现上是合理的：

1. **访存合并方向错位**。本实现的 kernel 每次读一组 `values[k]` + `col_idx[k]` + `x[col_idx[k]]`——前两者沿 CSR 的 NNZ 轴是连续的（相邻行的非零元相邻存储），理论上 256 个线程并行时 `values` 和 `col_idx` 读是合并的；但 `x[col_idx[k]]` 的地址由 `col_idx` 决定，合并性取决于 `col_idx` 在 warp 内是否连续。random 矩阵的 `col_idx` 在每行内本身就是随机的，但在 CSR 排布下"相邻行的相邻非零元"的列号是跳变的；banded 矩阵的 `col_idx` 是规则带，但在 warp 内不同行取到的列号也不连续——反而 random 因为"每行内列号随机"让 warp 级别的 cache 命中更均匀。

2. **空 warp 比例**。banded 每行固定 16 个元素，skewed 行长度重尾。随机矩阵虽然每行 16 个非零元的位置随机，但因为总行数大（131072）、行内非零元位置相对均匀分散，warp 内"线程 i 行非零元早结束而线程 j 行还在跑"的分叉相对 banded 更均衡。banded 规则行反而因为每行起点对齐、循环次数一致，在 warp 内让所有线程几乎同时到达 `end`，在一线程一行的基线里看不出优势。

3. **H2D / D2H 占比**。2M NNZ 的 CSR 存约 `2 × 2M × 4B + 2M × 4B ≈ 24 MB`，加上 `x`（512 KB）和 `y`（512 KB），H2D + D2H 合计约 50 MB。PCIe Gen4 在 L4 上能跑 ~12 GB/s，纯拷贝只需 4 ms 量级；实际端到端接近这个数，说明**端到端时间被内存传输主导，kernel 计算本身占比不高**，这是后续优化的首要目标——减少 H2D/D2H（但实验边界不允许全局缓存，只能在单次调用内做 kernel 合并或临时缓冲区）。

**skewed vs random（同为 131072 行，skewed 行长度重尾，random 行长度均匀）**：两者吞吐接近（skewed 0.74 vs random 0.80），因为基线实现里 skewed 只让少数长行被单个线程拖累，但 4096 行的重尾分布让这些长行覆盖了整个 kernel 时间的上限，而大量空/短行线程很快结束。random 虽然每行长度一致，但 warp 内 `col_idx` 的随机访问让 L2 cache 命中率更差——两者在 L4 的 32 MB L2 上互相抵消，最终吞吐率相差不到 8%。

**cage11（真实矩阵，39082 行，559722 NNZ）**：总数据量只有 banded 的约 1/4，但吞吐率只有 0.31 GNNZ/s，是所有矩阵中最低的。这是因为 cage11 的行长度分布和列访问模式都极不规则（真实电路仿真矩阵），`x[col_idx[k]]` 的访存几乎全是 L2 miss，且行内非零元数目差异大，warp 分支严重；同时矩阵规模小也让 SM 占用率下降，进一步拉低吞吐。

总体来看，一线程一行的基线在这些矩阵上吞吐率集中在 0.31–0.80 GNNZ/s 区间，瓶颈主要在**H2D/D2H 固定开销 + 非合并的 `x` 访存**。后续可探索的方向包括：多线程协作处理一行以隐藏长行延迟、向量化 `float4` 读取 CSR 值、用共享内存缓存小块 `x`、以及拆分为两个 kernel（先计算后并行归约）但受"端到端"计时影响要注意启动开销。

## 评分与提交

- 正确性与工程质量 70 分：构建/接口、基础与边界矩阵、随机/真实/长行/严重不均匀矩阵、重复调用、输入不变、输出完整覆盖、CUDA 错误处理和资源释放。
- 实验报告 10 分：按本模板记录真实环境、设计、检查结果、公开数据和分析。
- 性能 20 分：只对截止前隐藏正确性全部通过的有效学生提交排名。

开放期间只显示排名、12 个隐藏矩阵的综合 GNNZ/s 和累计提交次数，不显示性能分。截止后在 NVIDIA L4 上统一串行复测，十二项吞吐率的几何平均记为 $G_i$；全班最快有效学生提交记为 $G_{\max}^{student}$，性能分为

$$
P_i=20\frac{G_i}{G_{\max}^{student}}.
$$

补交没有性能分。提交完整仓库压缩包，不要只交源文件；不得包含 `build/`、`results/`、ZIP、缓存。
