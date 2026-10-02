# CUDA 矩阵乘法：分块与数据复用

矩阵乘法优化的核心，是让一次搬进来的数据参与更多计算。把全局内存、共享内存和寄存器的复用关系理清，许多优化就能放到同一个框架里理解。

## 先固定计算与存储约定

设 `A` 为 `M × K`，`B` 为 `K × N`，输出 `C` 为 `M × N`：

$$
C_{ij} = \sum_{k=0}^{K-1} A_{ik} B_{kj}
$$

`M、N` 是输出的两个方向，`K` 是累加方向。对连续的行优先存储，三个元素的地址分别是 `A[i * K + k]`、`B[k * N + j]`、`C[i * N + j]`。

最直观的映射是一个线程计算一个输出元素。线程负责输出的行列位置，沿 `K` 累加；整个 Grid 覆盖输出矩阵，而不是把全部输出都塞进一个线程块。

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc = fmaf(A[row * K + k], B[k * N + col], acc);
    }
    C[row * N + col] = acc;
}
```

这个片段容易建立正确性基线，但相邻输出重复使用的输入没有被显式组织起来。

## Block Tile 让一批线程共享输入

一个线程块负责 `BM × BN` 的输出区域，沿 `K` 每次处理 `BK`：

| 当前分块 | 大小 | 用途 |
| --- | --- | --- |
| A 的输入块 | `BM × BK` | 每个 A 元素可被多个输出列复用 |
| B 的输入块 | `BK × BN` | 每个 B 元素可被多个输出行复用 |
| C 的输出块 | `BM × BN` | 累加当前 K 分块的贡献 |

块内线程协作把输入放进共享内存，再使用它们进行乘加。加载完成后，消费者要确认数据就绪；进入下一个 K 分块前，也要保证当前输入不再被读取，才能覆盖缓冲区。

输出块的位置由 `blockIdx.y * BM` 和 `blockIdx.x * BN` 确定。Grid 大小对应 `ceil(N / BN)` 与 `ceil(M / BM)`，不能把输入分块尺寸直接当成 Grid 的线程块数量。

一个具体的量化例子：`BM = BN = 128`、`BK = 8` 时，当前分块加载 2048 个 float，完成 131072 次乘加。按一次乘加算 2 FLOPs、一个 float 算 4 字节，只计输入搬运，计算量与输入字节之比是 32 FLOPs/byte。这是复用关系的理想估算，还没有计入输出写回等成本。

## Thread Tile 让寄存器里的值继续复用

共享内存解决的是线程之间的复用。Thread Tile 则让每个线程持有 `TM × TN` 个输出累加器，并在每个 K 位置加载 `TM` 个 A 值、`TN` 个 B 值，完成一次外积累加。

例如 `TM = TN = 4`，读取 4 个 A 值与 4 个 B 值，可以更新 16 个输出累加器。相比重复为每个输出加载操作数，这些片段在寄存器里得到复用。

```cpp
for (int m = 0; m < TM; ++m) {
    for (int n = 0; n < TN; ++n) {
        acc[m][n] = fmaf(a_frag[m], b_frag[n], acc[m][n]);
    }
}
```

如果 `BM = BN = 128`、`TM = TN = 8`，线程块需要 `16 × 16 = 256` 个线程，每个线程持有 64 个输出值。更大的 Thread Tile 有更多复用机会，也会增加寄存器需求，甚至引入 spill；因此并非越大越好。

从全局内存搬入共享内存的线程映射，和每个线程计算哪块输出的映射，可以分别设计。两者都必须覆盖完整的数据范围，但不必使用同一种坐标排布。

## 共享内存分块的完整参考实现

先选一个容易验证的配置：T = 16，每个线程计算一个输出。块内所有线程都参与加载和同步，矩阵边缘用 0 填充，最后只写回合法输出位置。

```cpp
extern "C" __global__ void matmul_tiled(
    const float* A, const float* B, float* C, int M, int N, int K) {
    const int T = 16;
    __shared__ float As[16][16], Bs[16][16];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * T + ty;
    int col = blockIdx.x * T + tx;
    float acc = 0.0f;
    for (int base = 0; base < K; base += T) {
        As[ty][tx] = row < M && base + tx < K
                  ? A[row * K + base + tx] : 0.0f;
        Bs[ty][tx] = base + ty < K && col < N
                  ? B[(base + ty) * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < T; ++k)
            acc = fmaf(As[ty][k], Bs[k][tx], acc);
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
```


启动方式与这份固定配置对应：

```cpp
dim3 block(16, 16);
dim3 grid((N + 15) / 16, (M + 15) / 16);
matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
```

第一个同步点把生产者和消费者分开：A、B 小块写完以后，所有线程才能开始读取。第二个同步点把消费者和下一轮生产者分开：当前小块使用完以后，才能覆盖 As、Bs。不能为了屏蔽边缘输出，让部分线程在这两个屏障之前直接 return。

当 K = 33 时，循环处理基址 0、16、32 三个分块。最后一个分块只有 1 个有效 K 位置，其他位置填 0。这比假定所有矩阵维度都能被分块大小整除更适合建立正确性基线。

从输入读取量看，一个完整的 16×16×16 分块加载 512 个 float、计算 4096 次乘加。若每个输出独立从全局内存读取所有操作数，逻辑加载量是 8192 个 float。前者显式组织了输入复用；真实内存流量还会受到缓存、事务和输出写回影响，不能把逻辑加载比直接当成耗时比。

## Thread Tile 如何把一次加载变成多次乘加

下面把输出块扩大为 64×64，线程块仍为 16×16。每个线程持有 4×4 个输出累加器，所以 256 个线程覆盖 4096 个输出。

输入块的加载使用一维 tid 分配，计算则使用 threadIdx.x、threadIdx.y 定位线程的输出微块。这两个映射分开设计，正是“谁搬数据”和“谁计算结果”不必相同的具体实现。

```cpp
extern "C" __global__ void matmul_thread_tile(
    const float* A, const float* B, float* C, int M, int N, int K) {
    const int BM = 64, BN = 64, BK = 16, TM = 4, TN = 4;
    __shared__ float As[64][16], Bs[16][64];
    int tid = threadIdx.y * 16 + threadIdx.x;
    int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
    int local_row = threadIdx.y * TM, local_col = threadIdx.x * TN;
    float acc[4][4] = {};

    for (int base = 0; base < K; base += BK) {
        for (int j = tid; j < BM * BK; j += 256) {
            int m = j / BK, k = j % BK;
            As[m][k] = block_row + m < M && base + k < K
                       ? A[(block_row + m) * K + base + k] : 0.0f;
        }
        for (int j = tid; j < BK * BN; j += 256) {
            int k = j / BN, n = j % BN;
            Bs[k][n] = base + k < K && block_col + n < N
                       ? B[(base + k) * N + block_col + n] : 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < BK; ++k) {
            float a[4], b[4];
            for (int m = 0; m < TM; ++m) a[m] = As[local_row + m][k];
            for (int n = 0; n < TN; ++n) b[n] = Bs[k][local_col + n];
            for (int m = 0; m < TM; ++m)
                for (int n = 0; n < TN; ++n)
                    acc[m][n] = fmaf(a[m], b[n], acc[m][n]);
        }
        __syncthreads();
    }
    for (int m = 0; m < TM; ++m)
        for (int n = 0; n < TN; ++n) {
            int row = block_row + local_row + m;
            int col = block_col + local_col + n;
            if (row < M && col < N) C[row * N + col] = acc[m][n];
        }
}
```


对应启动配置为：

```cpp
dim3 block(16, 16);
dim3 grid((N + 63) / 64, (M + 63) / 64);
matmul_thread_tile<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
```

每个 k 位置，线程读取 4 个 A 值和 4 个 B 值，更新 16 个累加器。整个 K 方向完成前，累加器留在线程私有的寄存器逻辑中，而不是每次乘加都写回共享内存或全局内存。

这里的数组大小和循环界限是编译期常量，有利于编译器展开与分配寄存器；但它是否真正全部驻留寄存器，以及寄存器数量是多少，仍要看编译结果。本文的实现用于说明 Thread Tile 的机制，不代表完成了 Bank 布局、向量化和流水线等后续优化。

## 从访存瓶颈到寄存器压力

![原资料的内存访问分析图](/blogs/cuda-matmul-tiling/source-memory-chart.png)

Naive、共享内存分块和 Thread Tile 的区别，不只是总耗时不同，瓶颈也可能移动。显式复用减少了重复的全局加载，共享内存读取和指令发射可能因此变得更突出；线程内复用又减少了部分共享内存访问，同时扩大寄存器需求。

![原资料 Thread Tile 版本的启动统计](/blogs/cuda-matmul-tiling/source-register-pressure.png)

这张原始截图中 Registers Per Thread 为 141、Grid Size 为 64。它提示检查两种可能的限制：每个块消耗的寄存器是否限制驻留数量，以及 Grid 是否有足够多的块覆盖 GPU。不能仅凭寄存器数上升就判定优化失败，也不能仅凭访存请求减少就判定总体更快。

参考实现给出了机制与正确性边界。追求吞吐时，需要把输入形状、分块参数、寄存器分配、共享内存开销与实际 Kernel 时间放在一起看。

## cuBLAS 基线要先对齐布局与精度

经典 cuBLAS GEMM 接口使用列优先约定。连续的行优先缓冲区可视为相应转置矩阵的列优先表示，利用 `Cᵀ = Bᵀ Aᵀ`，可以交换 A、B 的传入顺序。

对连续行优先的 `A[M,K]`、`B[K,N]`、`C[M,N]`，一组对应参数是：

```cpp
// alpha = 1，beta = 0；指针模式为 host
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha,
            B, N, A, K, &beta, C, N);
```

这里交换的是解释布局与运算次序，没有把内存里的矩阵真的转置。若缓冲区有额外行间距，leading dimension 也必须随实际 stride 调整。[cuBLAS GEMM 文档](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm)

性能比较还要对齐数据类型、累加精度、数学模式与计时范围。“达到 cuBLAS 的某个百分比”只有放在具体输入和硬件条件下才有意义。

我把矩阵乘法的这条优化路线记成：先保证索引正确，再把共享输入放到共享内存，最后把线程需要反复使用的小片段留在寄存器中。真正重要的是每次搬运产生了多少可复用的计算。

## 参考实现的验证范围

本文补充的 CUDA 设备代码使用 CUDA NVRTC 12.9 编译，并在 NVIDIA GeForce RTX 5070 Laptop GPU 上与 CPU 参考结果对比。共享内存与 Thread Tile 实现检查了 (M,N,K) 为 (7,9,5)、(16,16,16)、(17,19,33)、(65,67,19) 的结果，包含非整除的矩阵边缘。

这组检查验证计算结果与所列边界，不是性能基准。正文的原资料截图仍用于分析机制，不能与本机正确性检查混成同一次实验。

[下载本文这组参考实现的 CUDA 源码](/blogs/cuda-matmul-tiling/core-kernels.cu)

## 学习来源

- [矩阵乘法的原始学习资料](https://tvle9mq8jh.feishu.cn/docx/KuY0dOEnFokWvKxnLlVcKo9Kngb)
- [CUDA：共享内存中的矩阵乘法](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab)
- [cuBLAS 文档](https://docs.nvidia.com/cuda/cublas/index.html)
- 延伸：[向量化访存与内存布局](/blog/cuda-vectorized-memory-access)
