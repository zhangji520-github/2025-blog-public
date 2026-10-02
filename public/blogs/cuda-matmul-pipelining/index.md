# CUDA 矩阵乘法流水线：双缓冲与 Warp 分块

分块和向量化解决了数据怎么搬、怎么复用。流水线进一步关注：下一批数据什么时候准备，能否让当前计算覆盖一部分等待时间。

## 双缓冲提供空间，流水线安排时间

单缓冲的逻辑顺序是“加载 tile 0 → 计算 tile 0 → 加载 tile 1 → 计算 tile 1”。双缓冲则保留两个输入区，在一个区的数据参与计算时，另一个区可准备下一批数据。

```text
填充：准备 tile 0
稳态：计算 tile q，同时准备 tile q+1
收尾：计算最后一个 tile
```

若一批数据的搬运耗时为 L、计算耗时为 T，串行模型是 `L + T`；理想重叠下，稳态可以接近 `max(L, T)`。这只是解释收益来源的时间模型，实际还要计入填充、收尾、同步和资源限制。

两个数组本身不会自动产生重叠。普通加载需要合理的预取与指令安排；异步拷贝还需要明确的提交、等待和缓冲区生命周期。是否真的隐藏等待，要由生成的指令与性能分析验证。

## 两层预取对应两种数据粒度

| 层级 | 准备的内容 | 当前正在做什么 |
| --- | --- | --- |
| 共享内存级 | 下一组 `BM × BK`、`BK × BN` 输入分块 | 使用当前 K 分块做乘加 |
| 寄存器级 | 下一个 k 位置的 A、B 小片段 | 累加当前片段的外积 |

共享内存级处理整块输入，寄存器级处理一个线程即将使用的小片段。二者可以结合，但不能把“预取下一轮操作数”与“写回计算结果”混为一谈。输出累加器通常在寄存器中累积整个 K 方向，最后才写回 C。

## 乒乓索引背后是缓冲区所有权

`current ^= 1` 只切换编号，不保证数据已经写完，也不保证上一轮消费者已经读完。关键约束有三个：

1. 读取一个缓冲区前，下一批输入必须完成写入并对消费者可见。
2. 覆盖一个缓冲区前，上一批消费者必须不再使用它。
3. 第一批数据要预先填充，最后一轮不能继续预取越界的 tile。

用语义伪代码表示：

```text
预载首个 tile，并确认它可读
对每个 tile：
    若存在下一批，向空闲缓冲区发起预取
    计算当前缓冲区里的数据
    确认下一批已就绪，当前批已被消费完
    交换读写角色
```

使用 `cuda::pipeline` 时，生产者 acquire/commit 与消费者 wait/release 分别描述这些阶段。异步拷贝的完成与线程间结果可见性，需要按实际参与线程和 API 语义处理，不能只保留索引翻转。[CUDA Pipelines 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html)

双缓冲还会消耗更多共享内存和寄存器。对 float 输入，两个缓冲阶段的 A、B 共享内存空间约为 `2 × BK × (BM + BN) × 4` 字节，尚未计 padding。若它限制了驻留块数量，隐藏延迟的收益可能被抵消。

## 一个明确表达缓冲区切换的实现

下面用 16×16 的输出 tile 表达软件预取：先把下一批输入读到线程寄存器，再计算当前共享内存中的输入，最后把预取值放入另一组共享内存缓冲区。

```cpp
extern "C" __global__ void matmul_double_buffer(
    const float* A, const float* B, float* C, int M, int N, int K) {
    const int T = 16;
    __shared__ float As[2][16][16], Bs[2][16][16];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
    As[0][ty][tx] = row < M && tx < K ? A[row * K + tx] : 0.0f;
    Bs[0][ty][tx] = ty < K && col < N ? B[ty * N + col] : 0.0f;
    __syncthreads();
    int current = 0;
    float acc = 0.0f;

    for (int base = 0; base < K; base += T) {
        int next_base = base + T;
        float next_a = row < M && next_base + tx < K
                       ? A[row * K + next_base + tx] : 0.0f;
        float next_b = next_base + ty < K && col < N
                       ? B[(next_base + ty) * N + col] : 0.0f;
        for (int k = 0; k < T; ++k)
            acc = fmaf(As[current][ty][k], Bs[current][k][tx], acc);
        __syncthreads();
        int next = current ^ 1;
        As[next][ty][tx] = next_a;
        Bs[next][ty][tx] = next_b;
        __syncthreads();
        current = next;
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
```


这份实现按 16×16 线程块启动，Grid 为 ceil(N/16)×ceil(M/16)，支持非整除的 M、N、K。每个线程写自己的合法 C 位置；非法输入位置填 0，所有线程仍然共同经过屏障。

它展示了全局内存→寄存器→下一共享内存缓冲区的生命周期。提前写下 next_a、next_b 不意味着硬件一定形成理想重叠，编译器的指令安排与实际等待仍需检查。它也不是把普通加载当成 cuda::memcpy_async；后者具有专门的异步提交与完成语义。

| 时点 | 当前缓冲区 | 另一个缓冲区或寄存器 |
| --- | --- | --- |
| 进入首轮前 | 首个 tile 已加载并可见 | 可用于下一批 |
| 计算当前 tile | 消费 As[current]、Bs[current] | next_a、next_b 保存后续输入 |
| 计算结束后 | 确认所有消费者结束 | 把预取值写入 As[next]、Bs[next] |
| 切换角色前 | 本轮输入可在未来被覆盖 | 下一轮输入已经可读 |

第一批需要填充，最后一批需要收尾。K 分块越少，这些额外步骤占总时间的比例越大；小矩阵没有明显收益，并不反驳流水线的原理，而是说明问题规模和开销之间的关系不同。

## 寄存器级双缓冲与共享内存级双缓冲

共享内存级的对象是一整组 K tile；寄存器级的对象是当前线程下一次 FMA 需要的 A、B 片段。把它们分开，能避免混淆“下一片段”和“下一大块”。

寄存器级的软件流水可以概括为：先加载片段 0；计算片段 k 的时候提前读取片段 k+1；随后翻转片段索引。最后一个片段不能继续访问超出 BK 的位置，跨共享内存 tile 时还要保证使用的是正确的大块缓冲区。

输出 acc 是另一类数据。它累积所有 K tile 的贡献，不能在每次交换输入缓冲区时清空，也通常不需要逐轮写回共享内存。输入缓冲区的复用与输出累加器的生命周期，是两条不同的线。

## Warp Tile 细化的是计算任务的归属

一个 `BM × BN` 输出块可以继续划分为多个 `WM × WN` 的 Warp Tile，再由每个 Warp 的线程计算更小的 Thread Tile。

例如输出块是 `128 × 128`，Warp Tile 是 `64 × 32`，就有 `2 × 4 = 8` 个 Warp，对应 256 个线程。若每线程负责 `8 × 8` 个输出值，一个 Warp 的 32 个线程正好覆盖 `64 × 32`。

```cpp
int warp_id = threadIdx.x / 32;
int warp_row = warp_id / (BN / WN);
int warp_col = warp_id % (BN / WN);
// Warp 输出区域起点：(warp_row * WM, warp_col * WN)
```

这些关系假设分块可整除。更复杂的实现可能让一个 Warp 多轮覆盖它的区域，但核心仍是给每个输出一个明确的线程归属。

Warp 原本就存在于线程块中。Warp Tiling 改变的是数据和任务的组织方式，不是凭空增加硬件线程，也不自动提高 occupancy。它的价值需要从访存模式、寄存器复用、调度与资源开销中判断。

## Warp 分块中的迭代维度

如果一个 Warp 一轮处理的区域比它负责的整个输出区域小，就需要保留原资料中的 WSUBM、WSUBN 与迭代维度，而不只记住 BM、BN、WM、WN。

例如一个 Warp 负责 64×32 个输出，每个线程一次更新 4×4 个值。32 个线程一轮可覆盖 32×16 的子区域，因此需要在 M、N 方向分别迭代 2 次。每个线程最终保存 2×2×4×4 = 64 个累加值。

| 层级 | 这个例子的范围 | 关系 |
| --- | --- | --- |
| Block | 128×128 | 包含 8 个 64×32 Warp Tile |
| Warp | 64×32 | 包含 2×2 个 32×16 子区域 |
| 单轮子区域 | 32×16 | 由 32 个线程覆盖 |
| 单线程单轮 | 4×4 | 每线程单轮更新 16 个值 |

约束可以写成：

$
\frac{WSUBM}{TM}\frac{WSUBN}{TN}=32,\qquad
WM=WMITER\cdot WSUBM,\qquad
WN=WNITER\cdot WSUBN
$

一个线程在 Warp Tile 内的行坐标，需要合并 Warp 起点、子区域迭代偏移和 lane 负责的微块偏移。例如 row = warp_row×WM + iter_m×WSUBM + lane_row×TM + m。列坐标使用对应的 N 方向关系。每一项都对应一个明确层级，比分散背诵索引表达式更容易检查重复与遗漏。

这个设计扩大了每个线程的输出覆盖范围，也扩大了累加器数量。能否换来更好的吞吐，需要结合寄存器压力、共享内存读取布局和并发驻留情况判断，不能只看“多加了一个 Warp 层级”。

## 如何阅读流水线后的调度器统计

![原资料双缓冲版本的调度器统计](/blogs/cuda-matmul-pipelining/source-scheduler.png)

图中 Active Warps Per Scheduler 约为 3.89，Eligible Warps 约为 1.83，Issued Warp 约为 0.68。Active 与 Eligible 的差异说明仍有一部分驻留 Warp 在等待；Eligible 与 Issued 的差异则说明已经就绪，也不意味着每个 Warp 都能在同一时刻被发出。

相对基准，Eligible 增加、No Eligible 减少，这支持“准备好下一批数据，减少无工作可发出的时段”的方向。但最终仍应检查 Kernel 时间，以及更多缓冲区占用是否降低了驻留数量。截图属于原资料的实验，数据用于解释指标之间的关系，不把它当成适用于所有矩阵大小的加速证明。

## 我会验证哪些变化

流水线优化后，最值得关注的是 Kernel 耗时、Eligible Warps、相关访存等待，以及寄存器和共享内存使用量。`Not Selected` 增加，可能说明就绪 Warp 已经足够，不能单凭名字把它当成坏事。[Nsight Compute 性能排查指南](https://docs.nvidia.com/nsight-compute/ComputeTriage/)

我把双缓冲理解为一种时间上的复用：当前操作数正在被消费时，下一批操作数提前准备。把“可读”和“可覆盖”的时点安排正确，才有资格讨论重叠带来的性能收益。

## 参考实现的验证范围

本文补充的 CUDA 设备代码使用 CUDA NVRTC 12.9 编译，并在 NVIDIA GeForce RTX 5070 Laptop GPU 上与 CPU 参考结果对比。双缓冲矩阵乘法检查了 (M,N,K) 为 (7,9,5)、(16,16,16)、(17,19,33)、(65,67,19) 的结果，包含首轮、尾轮与非整除 K。

这组检查验证计算结果与所列边界，不是性能基准。正文的原资料截图仍用于分析机制，不能与本机正确性检查混成同一次实验。

[下载本文这组参考实现的 CUDA 源码](/blogs/cuda-matmul-tiling/core-kernels.cu)

## 学习来源

- [流水线与 Warp 分块的原始学习资料](https://tvle9mq8jh.feishu.cn/docx/PtmBdbqktoiCfHxeStXc0KVbngh)
- [CUDA：Pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html)
- [Nsight Compute：Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/)
- 相关：[向量化访存与内存布局](/blog/cuda-vectorized-memory-access)
