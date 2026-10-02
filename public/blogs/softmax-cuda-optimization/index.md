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

## 学习来源

- [Softmax 相关学习资料](https://tvle9mq8jh.feishu.cn/docx/XLjZd9FVDoLXDGxJMhmcOvKwnrh)
- [NVIDIA：Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
- [NVIDIA：Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [NVIDIA：Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/)
