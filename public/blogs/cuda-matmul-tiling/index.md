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

## 学习来源

- [矩阵乘法的原始学习资料](https://tvle9mq8jh.feishu.cn/docx/KuY0dOEnFokWvKxnLlVcKo9Kngb)
- [CUDA：共享内存中的矩阵乘法](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab)
- [cuBLAS 文档](https://docs.nvidia.com/cuda/cublas/index.html)
- 延伸：[向量化访存与内存布局](/blog/cuda-vectorized-memory-access)
