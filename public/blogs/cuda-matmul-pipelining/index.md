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

## 我会验证哪些变化

流水线优化后，最值得关注的是 Kernel 耗时、Eligible Warps、相关访存等待，以及寄存器和共享内存使用量。`Not Selected` 增加，可能说明就绪 Warp 已经足够，不能单凭名字把它当成坏事。[Nsight Compute 性能排查指南](https://docs.nvidia.com/nsight-compute/ComputeTriage/)

我把双缓冲理解为一种时间上的复用：当前操作数正在被消费时，下一批操作数提前准备。把“可读”和“可覆盖”的时点安排正确，才有资格讨论重叠带来的性能收益。

## 学习来源

- [流水线与 Warp 分块的原始学习资料](https://tvle9mq8jh.feishu.cn/docx/PtmBdbqktoiCfHxeStXc0KVbngh)
- [CUDA：Pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html)
- [Nsight Compute：Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/)
- 相关：[向量化访存与内存布局](/blog/cuda-vectorized-memory-access)
