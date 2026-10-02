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

## 完整结果需要多层合并

我把通用的 Reduce 结构记成四层：

1. 线程在寄存器中累加自己的元素。
2. Warp 内通过 shuffle 合并局部值。
3. 各 Warp 把结果写入共享内存，合并成线程块结果。
4. 多个线程块的结果通过后续 Kernel 或其他明确的全局合并机制处理。

`__syncthreads()` 只同步一个线程块，不能充当整个 Grid 的屏障。常见的两次 Kernel 方案，可以在同一个 stream 中先生成块级结果，再归约这些结果。

最后一个 Warp 也不能依赖“天然同步”随意省掉共享内存通信的同步。使用 shuffle 时要满足参与 mask 的约束；通过共享内存交换数据时，需要明确的同步与内存顺序。[NVIDIA Warp 原语说明](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)

对我来说，归约优化最重要的判断是：减少每个元素需要的通信，让活跃线程的地址访问更规整，同时保证每一层的结果可见性。实际工程中，也值得先用 CUB 的归约实现建立正确性和性能基线。

## 学习来源

- [归约算子的原始学习资料](https://tvle9mq8jh.feishu.cn/docx/XoHCdUyGUoyLfmx3DZbcfhzAnfc)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [NVIDIA：Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
- [CCCL：Determinism](https://nvidia.github.io/cccl/unstable/cccl/determinism.html)
