# CUDA 向量化访存：指令宽度与内存布局

`float4` 的价值不在于让算法少读了数据，而在于用更宽的访存指令搬运连续元素。它要和线程映射、地址对齐及共享内存布局一起考虑。

## 向量化与合并访存是两个维度

| 优化 | 关注范围 | 关注的问题 |
| --- | --- | --- |
| 向量化访存 | 一个线程的一条指令 | 能否一次加载多个连续元素？ |
| 合并访存 | 一个 Warp 中各线程的地址 | 访问能否组成高效的内存事务？ |

一个线程一次读取 4 个 float，不代表 Warp 的访问自动最优；相邻线程读相邻地址，也不代表每个线程使用了宽指令。两种优化可以配合，但不能相互替代。

在满足对齐与边界条件、编译器确实生成宽访存指令时，4 次标量加载可能变为一次 16 字节加载。要搬运的总字节数不变，硬件事务数量也不能直接按“四分之一”推断。

```cpp
// ptr 已按 16 字节对齐，而且至少有 4 个有效 float
float4 values = *reinterpret_cast<const float4*>(ptr);
```

## 加载宽度会改变线程与数据的对应关系

假设要搬运一个 `8 × 8` 的 tile，有 8 个线程，每次读取 4 个连续元素。每行需要 2 个线程，一轮能搬运 4 行，所以需要两轮。

| 线程 | 第一轮 | 第二轮 |
| --- | --- | --- |
| thread 0 | 第 0 行，第 0～3 列 | 第 4 行，第 0～3 列 |
| thread 1 | 第 0 行，第 4～7 列 | 第 4 行，第 4～7 列 |
| thread 2 | 第 1 行，第 0～3 列 | 第 5 行，第 0～3 列 |

对行宽为 `BK` 的 A 分块，向量宽度为 4，一种映射是：

```cpp
int groups_per_row = BK / 4;
int row = tid / groups_per_row;
int col = (tid % groups_per_row) * 4;
int row_stride = blockDim.x / groups_per_row;
```

这组简化关系假设 `BK` 能被 4 整除、线程数能组成完整的搬运行。后续轮次按 `row_stride` 移动；尾块与无法整除的尺寸需要单独处理。

加载映射描述谁搬哪些输入，计算映射描述谁负责哪些输出。不能因为一个线程计算 `TM × TN` 个结果，就认定它必须从全局内存搬运相同数量的输入。

## 转置共享内存布局是为了服务消费者

全局内存中的 A 分块通常按 `BM × BK` 的行优先顺序读取，适合连续加载。计算时，一个线程可能需要固定 K 位置、读取多个 M 方向的值。

如果共享内存保留原布局，访问 `As[m * BK + k]` 时，相邻 m 的地址相差 `BK`。把它写成转置布局后，变为 `AsT[k * BM + m]`，同一 k 下的多个 m 值连续，便于后续宽加载。

```cpp
// 逻辑上仍是 A[m, k]，这里只调整共享内存中的存储位置
AsT[k * BM + m] = A_tile[m * BK + k];
```

这不是改变矩阵乘法的数学含义，而是让生产者按适合搬运的方式读取，让消费者按适合计算的方式访问。

共享内存是显式管理的存储，不应直接把这项收益归结为“缓存命中率更高”。仍要检查同一 Warp 的地址是否产生 Bank 冲突：转置可能改善一种访问，同时使另一种访问需要 padding 或其他布局调整。[NVIDIA 矩阵转置与 Bank 冲突分析](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)

## 连续加载与转置写入怎样连接起来

把生产者和消费者连起来看，会更容易理解为什么“全局内存连续读”与“共享内存换布局”可以同时成立。

下面是一个完整 64×8 tile 的参考搬运 Kernel。128 个线程各加载 4 个 float，输入地址按 16 字节对齐，最终写出一个 8×64 的转置结果：

```cpp
// One 64x8 complete tile, 128 threads, input is 16-byte aligned.
extern "C" __global__ void vector_transpose_tile(
    const float* input, float* output) {
    __shared__ float AsT[8][64];
    int tid = threadIdx.x;
    int row = tid / 2, col = (tid % 2) * 4;
    float4 v = *reinterpret_cast<const float4*>(input + row * 8 + col);
    AsT[col + 0][row] = v.x;
    AsT[col + 1][row] = v.y;
    AsT[col + 2][row] = v.z;
    AsT[col + 3][row] = v.w;
    __syncthreads();
    for (int j = tid; j < 8 * 64; j += 128)
        output[j] = AsT[j / 64][j % 64];
}
```


启动配置固定为一个 128 线程块，输入和输出分别至少容纳 512 个 float：

```cpp
vector_transpose_tile<<<1, 128>>>(d_input, d_transposed);
```

thread 0 加载 input 的第 0 行第 0～3 列，分别写入 AsT 的第 0～3 行第 0 列。thread 1 加载同一行第 4～7 列，写入 AsT 的第 4～7 行第 0 列。加载时连续的四个值，写入后成为四个不同 K 位置的同一个 M 位置；后续计算固定 K 时，沿 M 读取就能获得连续数据。

这份实现刻意限制为完整、对齐的 tile，用于验证坐标关系。通用 GEMM 需要处理矩阵尾部与行间距。例如行优先矩阵的 K 不是 4 的倍数时，即使 cudaMalloc 返回的基地址足够对齐，也不能保证每行的起始地址都满足 float4 的 16 字节对齐要求。

| 情况 | 可以采用的加载路径 |
| --- | --- |
| 起始地址对齐，剩余元素至少 4 个 | 用 float4 连续加载 |
| 起始地址不满足对齐 | 使用标量或其他已满足对齐的路径 |
| 尾部只有 1～3 个元素 | 按有效元素逐个加载，其余填中性值 |
| 输出行尾不足 4 列 | 逐个写回合法输出，不能宽写越界 |

一个数据加载分支可以根据这些条件选择路径；但同一块中的线程仍需按整体设计参与共享内存同步，不能让尾部线程直接跳过后续屏障。

## 对齐、转置与 Bank 布局需要一起权衡

普通的 32×32 共享内存转置，按列访问容易让多个 lane 落在同一个 Bank。经典的 32×33 padding 可以改变 Bank 映射，但行长变成 33 个 float 后，相邻行首的字节步长为 132，并不是 16 的倍数。

因此，某种 padding 适合标量的 Bank 布局，不代表它也适合直接对每一行使用 float4。设计宽访存布局时，需要同时算出每个线程的地址、类型对齐，以及每条共享内存指令的 Bank 映射。只有“转置”和“向量化”两个名字，不能决定最优布局。

## 结合真实分析图判断向量化的收益

![原资料向量化版本的访存统计](/blogs/cuda-vectorized-memory-access/source-vector-memory.png)

图中 Global 指令数与请求数相对基准下降 75%，符合“更宽的加载减少指令”的方向；但 L1/TEX 命中率约为 10.34%，相对基准下降。这个具体例子说明，不能把该优化统一解释成“缓存命中率提高”。命中率的变化要结合请求模式和实际流量解释。

![原资料的计算和内存吞吐指标](/blogs/cuda-vectorized-memory-access/source-throughput.png)

另一个图中 Compute 与 Memory 的吞吐指标同时上升。读图时还要区分 Memory Throughput 的汇总指标、DRAM 带宽、缓存与指令发射等具体资源；不能仅因为一个汇总百分比高，就断定已经是 DRAM 带宽受限。

我更愿意保留这样的分析顺序：先确认生成了什么宽指令，再确认请求与实际字节流量如何变化，随后检查寄存器和 Bank 冲突，最后用同一输入下的 Kernel 时间判断是否值得采用。这里引用的截图属于原资料的实验，不是本机的测速结果。

## 宽加载的前提和代价

- 地址满足向量类型的自然对齐要求；`float4` 对应 16 字节对齐，子矩阵偏移可能打破它。
- 尾部至少还有 4 个有效元素，不能让宽加载跨出合法范围。
- 输出写回同样要检查地址、行间距与剩余列数，不能只优化输入。
- 更多预加载值会占用寄存器；更少的指令不一定能抵消寄存器压力和较低占用率。

我的判断方式是：先确认访问地址正确，再看生成的指令、实际请求和资源使用，最后看 Kernel 耗时。向量化是一种改善数据搬运组织的方式，不是固定倍数的加速承诺。

## 参考实现的验证范围

本文补充的 CUDA 设备代码使用 CUDA NVRTC 12.9 编译，并在 NVIDIA GeForce RTX 5070 Laptop GPU 上与 CPU 参考结果对比。向量化搬运实现逐元素检查了完整 64×8 输入到 8×64 输出的转置关系。它的固定配置检查不包含通用矩阵尾部加载。

这组检查验证计算结果与所列边界，不是性能基准。正文的原资料截图仍用于分析机制，不能与本机正确性检查混成同一次实验。

[下载本文这组参考实现的 CUDA 源码](/blogs/cuda-matmul-tiling/core-kernels.cu)

## 学习来源

- [向量化访存的原始学习资料](https://tvle9mq8jh.feishu.cn/docx/GpIkdSRagoQur1xmbIYclR2Xngf)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [NVIDIA：An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
- 相关：[矩阵乘法的分块与数据复用](/blog/cuda-matmul-tiling)
