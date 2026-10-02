# CUDA 归约笔记：线程利用、Bank 冲突与分层合并

归约把一组数据合并为一个结果，例如求和、最大值或均值。值得复用的不是某个版本的代码，而是如何组织局部计算、线程间通信和跨块合并。

## 树形归约减少的是依赖深度

串行求和的结果依赖一路向前传递。树形归约可以把独立的加法并行展开：

```text
[1, 2, 3, 4, 5, 6, 7, 8]
       ↓ 两两合并
      [3, 7, 11, 15]
       ↓ 两两合并
         [10, 26]
       ↓ 合并
            36
```

8 个元素仍然需要 7 次加法，但依赖链从 7 层变成 3 层。一般情况下，总工作量是 `O(N)`，理想并行深度是 `O(log N)`，不能把两者混成“总计算量降到了对数级”。

浮点加法也不严格满足结合律。归约顺序改变后，低位结果可能不同，所以比较浮点结果需要误差容限，而不是只做逐位相等判断。[CCCL 关于浮点归约确定性的说明](https://nvidia.github.io/cccl/unstable/cccl/determinism.html)

## 让线程先做局部累加，再参与通信

只让每个线程读取一个数，随后立刻进入归约，很多线程很快就不再参与有效计算。一个常用做法是让每个线程先处理多个元素，在寄存器中累加，再把局部结果交给块内归约。

```cpp
float local_sum = 0.0f;
int start = blockIdx.x * blockDim.x + threadIdx.x;
int step = blockDim.x * gridDim.x;
for (int i = start; i < count; i += step) {
    local_sum += input[i];
}
```

同一轮中，相邻线程读相邻元素；后续轮次按整个 Grid 的步长继续读取。这样把“读取一个元素”变成“处理一组元素”，降低了每个输入元素对应的通信开销。

另一个例子是每个线程读取 `i` 和 `i + B` 两个位置后先求和。这里读的是两个输入区间，**并不是两个正在运行的线程块互相通信**。线程负责多个元素，与块之间同步，是不同的问题。

## 活跃线程的排布比判断式更重要

`tid % (2 * stride) == 0` 会让同一 Warp 中工作的线程越来越稀疏。改用顺序寻址，把活跃线程集中到前半部分，可以让前面的完整 Warp 持续工作，后面的 Warp 整体跳过。

典型的共享内存求和骨架是：

```cpp
// B 是 2 的幂；shared 已保存 B 个线程的局部和
for (int stride = B / 2; stride > 0; stride /= 2) {
    __syncthreads();
    if (tid < stride) {
        shared[tid] += shared[tid + stride];
    }
}
// tid == 0 的线程写出该块的结果
```

这里的收益来自线程参与模式和共享内存访问模式更规整，不只是“把取模换成比较”。同时，树形合并越接近末尾，活跃线程越少，这是归约结构本身的特点。

## Bank 冲突要看同一条访存指令

对常见的 32 位共享内存访问，可以用 `bank = word_index % 32` 理解映射：

| 同一个 Warp 的访问方式 | 对应现象 |
| --- | --- |
| lane 0～31 读取 `shared[0]`～`shared[31]` | 分散到不同 Bank |
| lane t 读取 `shared[t * 32]` | 不同地址集中到同一个 Bank，发生冲突 |
| 多个 lane 读取同一个地址 | 可以广播，不能简单当成 Bank 冲突 |

不能因为同一个线程先后访问两个同 Bank 的地址，就认定发生冲突。需要看同一 Warp 中各线程在一次访问请求中的地址分布。[CUDA 共享内存与 Bank 说明](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks)

## 把两种线程参与方式放在同一组数据上看

![间隔线程参与的归约过程](/blogs/cuda-reduction-notes/interleaved-reduction.png)

![活跃线程集中在前半部分的归约过程](/blogs/cuda-reduction-notes/sequential-reduction.png)

两张图来自学习资料，最终合并的是同一组值。它们的区别是哪些线程负责这些加法：间隔寻址让每个 Warp 内留下零散的有效 lane；顺序寻址让工作尽量集中到连续的线程范围。总加法数量没有因此改变，但活跃 lane 的组织方式和共享内存访问模式改变了。

以一个 256 线程的块为例，跨度从 128 开始时，前 4 个完整 Warp 工作，其余 Warp 整体跳过；跨度变成 64、32 时，分别还有 2 个、1 个完整 Warp 工作。只有进一步缩小到 16、8 等跨度，才在最后一个 Warp 内出现部分 lane 参与的情况。

“后半线程闲置”是树形合并后期不可避免的一部分。优化更有价值的地方，是先让每个线程在加载阶段累加更多输入，以及避免让多个 Warp 同时只剩很少的有效 lane。

## 完整结果需要多层合并

我把通用的 Reduce 结构记成四层：

1. 线程在寄存器中累加自己的元素。
2. Warp 内通过 shuffle 合并局部值。
3. 各 Warp 把结果写入共享内存，合并成线程块结果。
4. 多个线程块的结果通过后续 Kernel 或其他明确的全局合并机制处理。

`__syncthreads()` 只同步一个线程块，不能充当整个 Grid 的屏障。常见的两次 Kernel 方案，可以在同一个 stream 中先生成块级结果，再归约这些结果。

最后一个 Warp 也不能依赖“天然同步”随意省掉共享内存通信的同步。使用 shuffle 时要满足参与 mask 的约束；通过共享内存交换数据时，需要明确的同步与内存顺序。[NVIDIA Warp 原语说明](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)

对我来说，归约优化最重要的判断是：减少每个元素需要的通信，让活跃线程的地址访问更规整，同时保证每一层的结果可见性。实际工程中，也值得先用 CUB 的归约实现建立正确性和性能基线。

## 一个带尾部处理的两阶段 Reduce

先在线程中做 grid-stride 累加，再调用块级归约。辅助函数与 [Softmax 的核心实现](/blog/softmax-cuda-optimization) 相同：完整 Warp 内合并，Warp 结果写入共享内存，第一个 Warp 再做最终合并并广播。

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


```cpp
extern "C" __global__ void reduce_sum(
    const float* input, float* partial, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    float local = 0.0f;
    for (; i < count; i += step) local += input[i];
    float total = block_reduce(local, 0);
    if (threadIdx.x == 0) partial[blockIdx.x] = total;
}
```


启动时，第一次写出 P 个块级结果，第二次把它们归约成一个数。两个 Kernel 放在同一个 stream 中，第二次执行会在前一次结束后开始：

```cpp
int B = 128;
int P = 128; // 正数；实际可结合输入规模与硬件调节
reduce_sum<<<P, B, 0, stream>>>(d_input, d_partials, count);
reduce_sum<<<1, B, 0, stream>>>(d_partials, d_result, P);
// 读取 d_result 到主机前，等待 stream 完成并检查 CUDA 错误
```

d_partials 至少容纳 P 个 float，d_result 至少容纳一个 float，不能与仍被读取的输入重叠。count 不必是 B 的倍数：没有对应元素的线程保持 local = 0，仍然参与归约和同步。

以 count = 1048576、P = 128、B = 128 为例，每个线程平均累加 64 个输入值；共享内存里只交换每个 Warp 的一个结果。第二阶段只有 128 个输入值，通信规模已经很小。

## 如何解释性能分析中的“线程利用率”

![原资料中初始 Reduce 实现的 Warp 统计](/blogs/cuda-reduction-notes/source-warp-statistics.png)

图中的 Non Predicated Off Threads Per Warp 约为 23.75，它是采样或统计范围内的平均有效线程数，并不表示“每个时刻固定只有 23.75 个线程工作”。它可以提示检查谓词和控制流，但还要结合具体的执行指令和参与范围。

我会把这类证据与三件事对应：归约循环是否让 lane 稀疏地参与；共享内存地址是否发生冲突；最后一个 Warp 是否能用正确的 shuffle 通信缩短共享内存访问与同步路径。

计时也要说明范围。单次第一阶段 Kernel 时间，不等于两阶段 Reduce 的总时间，更不等于包含主机设备拷贝的端到端时间。比较时要把输入规模、数据类型、预热次数和这些边界对齐。

## 参考实现的验证范围

本文补充的 CUDA 设备代码使用 CUDA NVRTC 12.9 编译，并在 NVIDIA GeForce RTX 5070 Laptop GPU 上与 CPU 参考结果对比。两阶段求和检查了 1、255、256、257、8193 个元素，覆盖了线程块边界和非整除尾部。

这组检查验证计算结果与所列边界，不是性能基准。正文的原资料截图仍用于分析机制，不能与本机正确性检查混成同一次实验。

[下载本文这组参考实现的 CUDA 源码](/blogs/cuda-matmul-tiling/core-kernels.cu)

## 学习来源

- [归约算子的原始学习资料](https://tvle9mq8jh.feishu.cn/docx/XoHCdUyGUoyLfmx3DZbcfhzAnfc)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [NVIDIA：Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
- [CCCL：Determinism](https://nvidia.github.io/cccl/unstable/cccl/determinism.html)
