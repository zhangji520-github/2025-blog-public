# Softmax 优化笔记：归约、访存与并行度

整理 Softmax 时，我最想留下的是一个思路：公式没有变，性能差异往往来自数据如何分给线程，以及中间结果如何合并。

## Softmax 的核心是两次归约

对一行输入，稳定的计算形式是：

$$
m = \max_j x_j, \qquad s = \sum_j e^{x_j-m}, \qquad y_i = \frac{e^{x_i-m}}{s}
$$

这里有两次归约：求整行最大值，再求指数和。最后每个元素独立完成归一化。

减去最大值不会改变结果，却能避免大正数的指数溢出。例如输入 `[1000, 1001, 1002]`，先平移为 `[-2, -1, 0]`，得到的概率约为 `[0.0900, 0.2447, 0.6652]`。最大的平移值是 **0**，对应的指数值是 **1**。

一个容易忽略的依赖是：所有线程必须先得到同一个行最大值，才能直接累加可合并的指数和。如果各线程分别用自己的局部最大值计算指数，局部和不能直接相加。

## 数值平移为什么不改变概率

把分子和分母都除以同一个常数 exp(m)，比值不会改变。这是减去最大值的依据，而不只是一个经验上的“防溢出技巧”。

$
\frac{e^{x_i}}{\sum_j e^{x_j}} =
\frac{e^{x_i}/e^m}{\sum_j e^{x_j}/e^m} =
\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}
$

对应的 CPU 基线可以保持三段逻辑清晰：先求最大值，再计算指数和，最后除以分母。GPU 版本改变的是这三段工作如何分配给线程，数学依赖仍然相同。

| 阶段 | 一个线程可以独立完成的工作 | 进入下一阶段前需要什么 |
| --- | --- | --- |
| 最大值 | 求自己负责元素的局部最大值 | 整行的统一最大值 m |
| 指数和 | 计算 exp(x-m)，累加局部和 | 整行的分母 s |
| 归一化 | 把自己的指数项除以 s | 不再需要归约 |

这个顺序也解释了为什么不能随意混合局部最大值和局部和。若不同线程使用不同的 m，局部指数和的尺度不同；直接相加会改变结果。需要先统一最大值，或者使用另一个带尺度修正的归约算法。

## 线程映射决定了访存方式

把输入看成行优先存储的 `N × C` 矩阵，一个常见方案是一行分给一个线程块。块内有 `B` 个线程，每个线程跨步处理若干元素：

```cpp
int row = blockIdx.x;
int tid = threadIdx.x;
const float* x = inp + row * C;
for (int j = tid; j < C; j += blockDim.x) {
    // 当前线程处理 x[j]
}
```

以 `C = 4096`、`B = 128` 为例，每个线程处理 32 个元素。线程 0 读取 `0, 128, 256, …`，线程 1 读取 `1, 129, 257, …`。同一轮迭代中，相邻线程访问相邻地址，有利于合并访存。

这也解释了两个不同的并行维度：`N` 提供行与行之间的并行度，`B` 提供一行内部的并行度。只给每行一个线程，即使放到 GPU 上，行内的最大值与求和仍然是串行工作。

## 归约可以分成线程、Warp、线程块三层

| 组织方式 | 中间结果如何合并 | 主要取舍 |
| --- | --- | --- |
| 共享内存归约 | 各线程写入局部结果，再在块内做树形合并 | 容易理解，但需要共享内存访问和块级同步 |
| 每行一个 Warp | 32 个线程用 shuffle 合并寄存器中的结果 | 减少块级通信，但每行可用的线程数较少 |
| 多 Warp 协作 | Warp 内用 shuffle，Warp 之间用共享内存 | 兼顾行内并行度与通信成本 |

共享内存的树形归约，每轮把比较跨度减半。16 个局部结果对应 `8 → 4 → 2 → 1`，4 轮得到一个结果。常见的直接减半写法要求参与归约的数组长度是 2 的幂；换成任意长度时，需要处理多出来的元素。

Warp 内可以直接交换线程寄存器中的值。下面的片段适用于完整的 32 个 lane 都参与的情况，最终最大值由 lane 0 持有：

```cpp
__device__ float warp_max(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffffu, value, offset);
        value = fmaxf(value, other);
    }
    return value;
}
```

把最大值操作改为加法，就得到同样结构的求和归约。`mask` 表示参与线程，读取的源 lane 也必须参与；部分线程提前退出时，不能直接套用完整 Warp 的写法。shuffle 只负责 Warp 内的数据交换，也不能替代跨 Warp 共享内存通信所需的同步。[NVIDIA Warp 原语说明](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)

![Warp 内树形归约示意图](/blogs/softmax-cuda-optimization/figure-11.png)

图中用 8 个元素展示跨度 `4 → 2 → 1` 的合并过程，来源见文末学习资料。

多 Warp 方案的思路是：线程先处理自己的元素，Warp 内归约得到一个局部结果，再把各 Warp 的结果写入共享内存，合并成整行结果。对于 `B = 128`，块内共有 4 个 Warp，跨 Warp 阶段只需要合并 4 个值。

```cpp
int warp_id = threadIdx.x / 32;
int lane_id = threadIdx.x % 32;
if (lane_id == 0) {
    warp_maxima[warp_id] = local_warp_max;
}
__syncthreads();
// 一个线程合并各 Warp 的最大值，随后再次同步并广播结果
```

Warp 大小是 32，使用 shuffle 不代表整个线程块只能有 32 个线程。线程块可以包含多个完整 Warp，只是 shuffle 本身不能跨 Warp 传值。

## 一个能串起三层归约的核心实现

下面使用一维线程块，线程数 B 是 32 的倍数且不超过 1024。输入是有限的 FP32 数值，C 大于 0。完整 Warp 中没有分到元素的线程仍然参与归约：最大值使用有限 FP32 的最小值作初值，求和使用 0。

先把 Warp 归约和块级广播放在同一个辅助实现里：

```cpp
__device__ float warp_reduce(float value, int use_max) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffffu, value, offset);
        value = use_max ? fmaxf(value, other) : value + other;
    }
    return value; // lane 0 holds this Warp's result
}

// One-dimensional block, B is a multiple of 32, 32 <= B <= 1024.
// Every thread in the block must call this function.
__device__ float block_reduce(float value, int use_max) {
    __shared__ float partial[32];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    int warps = blockDim.x / 32;
    float neutral = use_max ? -3.402823466e38f : 0.0f;
    value = warp_reduce(value, use_max);
    if (lane == 0) partial[warp] = value;
    __syncthreads();

    if (warp == 0) {
        float v = lane < warps ? partial[lane] : neutral;
        v = warp_reduce(v, use_max);
        if (lane == 0) partial[0] = v;
    }
    __syncthreads();
    float result = partial[0];
    __syncthreads(); // all readers finish before the next reuse
    return result;
}
```


第一个同步点保证各 Warp 的结果已写入共享内存。第一个 Warp 读取这些结果，不足 32 个的位置填入中性值，然后再次归约。第二个同步点保证块级结果可读；最后一个同步点保证共享存储再次使用前，旧结果已经被所有线程读取。

这个辅助函数必须由整个线程块共同调用，不能只在 tid == 0 的分支里调用。否则其他线程到不了块级屏障，程序可能挂起。

有了这个共同的通信结构，Softmax 主体可以完整表达为：

```cpp
extern "C" __global__ void softmax_rows(
    const float* input, float* output, int N, int C) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= N) return; // uniform decision for the entire block
    const float* x = input + row * C;
    float* y = output + row * C;

    float local_max = -3.402823466e38f;
    for (int j = tid; j < C; j += blockDim.x)
        local_max = fmaxf(local_max, x[j]);
    float maximum = block_reduce(local_max, 1);

    float local_sum = 0.0f;
    for (int j = tid; j < C; j += blockDim.x) {
        float e = expf(x[j] - maximum);
        y[j] = e;
        local_sum += e;
    }
    float denominator = block_reduce(local_sum, 0);
    for (int j = tid; j < C; j += blockDim.x)
        y[j] /= denominator;
}
```


每个线程写入的输出位置和最后读取的位置相同，所以指数暂存到 output 后，归一化阶段不会读取其他线程尚未完成的输出值。跨线程交换的是 maximum 与 denominator；这些值由块级归约同步后广播。

启动配置可选 B = 128，Grid 有 N 个块。辅助函数使用静态共享内存，不需要再传额外的动态共享内存字节数：

```cpp
softmax_rows<<<N, 128>>>(d_input, d_output, N, C);
```

这是一种易于检查的实现，不是性能最优的结论。它把指数项暂存到全局输出，再读回做归一化；行较短时，可以考虑把更多数据留在寄存器中，但要同时观察寄存器压力。

![共享内存树形归约的数据合并过程](/blogs/softmax-cuda-optimization/shared-tree.png)

图来自学习资料。它把“每个线程持有局部值、逐轮合并到一个结果”的过程展开，帮助区分局部计算与线程间通信。

## 从一条等待指令追踪性能瓶颈

![原资料中的 Warp 停顿统计](/blogs/softmax-cuda-optimization/source-stalls.png)

原资料的分析截图用于说明排查方法，不能直接作为本机性能结果。Long Scoreboard 说明有 L1TEX 操作的依赖尚未完成，但还需要沿源码与 SASS 的依赖链追踪：哪条加载产生了寄存器值，哪条消费指令在等待它，是否有足够独立的工作覆盖等待。

当最大值与指数和的归约都已经缩短，下一步往往不是继续删一个同步点，而是确认输入的重复读取、输出暂存、线程工作量和寄存器占用，哪个环节限制了实际吞吐。

## 性能分析先区分并行度和等待

减少通信操作不一定更快。每行只有一个 Warp 时，通信成本较低，但行很长时，每个线程承担的工作也会更多。多 Warp 增加行内并行度，同时增加线程和寄存器等资源的需求。

因此，`block_size` 需要结合行宽、行数和硬件测试。更大的线程块可能减轻单线程负担，也可能降低同时驻留的线程块数量，不能简单认为 1024 一定比 128 快。

看 Nsight Compute 时，我会把指标和问题对应起来：

| 观察 | 需要继续确认的问题 |
| --- | --- |
| SM 和显存吞吐都低 | Grid 是否太小，工作量是否足够？ |
| Active Warps 多，Eligible Warps 少 | 驻留线程束是否在等待依赖或同步？ |
| Long Scoreboard 停顿明显 | 哪条 L1TEX 相关访存指令的依赖没有完成？ |
| 增大线程块后耗时反而增加 | 寄存器、共享内存或驻留块数量是否成为限制？ |

Active 表示驻留，Eligible 表示已准备好发出指令，两者不能混为一谈。Long Scoreboard 指向 L1TEX 操作的依赖等待，也不能只凭这一项就判断是某个同步函数造成的。[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

占用率与吞吐量要一起看：有更多驻留 Warp，并不自动意味着更高性能。[Nsight Compute 性能排查指南](https://docs.nvidia.com/nsight-compute/ComputeTriage/)

## 我留下的几个判断

- 先保证数值正确：所有线程共享行最大值，再计算和合并指数和。
- 先把数据分好，再选择归约方式：相邻线程的访问模式和每个线程的工作量同样重要。
- 通信有层级：Warp 内用 shuffle，跨 Warp 的结果需要另行合并与同步。
- 性能结论要带条件：比较同一输入、硬件、数据类型和计时范围，不能把某次加速比当作普遍结论。

这套思路可以延伸到求和、最大值、归一化等算子：把局部计算、结果合并和数据等待分开看，优化方向就更清楚。

## 参考实现的验证范围

本文补充的 CUDA 设备代码使用 CUDA NVRTC 12.9 编译，并在 NVIDIA GeForce RTX 5070 Laptop GPU 上与 CPU 参考结果对比。Softmax 检查了 3 行输入、行宽 1、31、32、33、257、4096 的结果，覆盖了小于一个 Warp、Warp 边界与长行的情况。

这组检查验证计算结果与所列边界，不是性能基准。正文的原资料截图仍用于分析机制，不能与本机正确性检查混成同一次实验。

[下载本文这组参考实现的 CUDA 源码](/blogs/cuda-matmul-tiling/core-kernels.cu)

## 学习来源

- [Softmax 相关学习资料](https://tvle9mq8jh.feishu.cn/docx/XLjZd9FVDoLXDGxJMhmcOvKwnrh)
- [NVIDIA：Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
- [NVIDIA：Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [NVIDIA：Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/)
