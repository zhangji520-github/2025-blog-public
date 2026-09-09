# 什么是 CUDA？从 GPU、SM 到线程束

作者：🦊大侠 · Fox Infra

## 1. CUDA 和 GPU 是什么关系？

**CUDA 是 NVIDIA 推出的并行计算平台和编程模型；GPU 是执行计算的硬件。** 使用 CUDA，我们可以让 NVIDIA GPU 执行图形渲染之外的通用计算，例如向量运算和深度学习中的张量计算。[NVIDIA CUDA](https://developer.nvidia.com/cuda)

因此，“CUDA 由多个 SM 组成”需要改成：**NVIDIA GPU 的计算架构包含多个 SM，CUDA 程序的线程在这些 SM 上执行。**

几个容易混淆的名字：

| 名称 | 含义 |
| --- | --- |
| GPU | 实际运行并行计算的处理器 |
| CUDA | 使用 NVIDIA GPU 进行通用并行计算的平台与编程模型 |
| CUDA Toolkit | 开发工具包，包括编译器、运行时和相关开发工具 |
| CUDA Core | GPU 内部的基本运算单元，不等于一个 CPU 核心 |
| Kernel（核函数） | 由许多 GPU 线程执行的一段函数代码，不是硬件核心 |

可以把 GPU 理解为工厂，SM 是车间，执行单元是不同工位。CUDA 提供安排并行任务的编程方式；这些比喻只帮助理解，不能当作严格的一一对应关系。

## 2. SM 内部有哪些部件？

SM 的全称是 **Streaming Multiprocessor，流式多处理器**。它把计算、访存、调度和片上存储资源组织在一起，为线程提供执行环境。

下面整理了课程中的主要部件，并补充了理解时需要注意的边界：

| 部件 | 主要作用 | 理解要点 |
| --- | --- | --- |
| CUDA Core / 运算流水线 | 执行加法、乘法等基本运算 | 不同架构对浮点、整数等指令的执行资源划分不同，不能把所有运算都理解成由同一种单元完成 |
| LD/ST（Load/Store） | 执行加载和存储相关指令 | 它是访存执行单元，不是存放数据的显存本身 |
| SFU（Special Function Unit） | 加速部分特殊数学运算 | 某些三角、指数、倒数相关指令可使用此类单元；具体函数可能被编译成多条指令，取决于架构、精度要求和编译选项 |
| Warp Scheduler | 从可执行的 warp 中选择要发射的指令 | 是否就绪取决于数据依赖、访存等待和执行资源等条件 |
| Dispatch Unit | 将选中的指令分派到对应执行流水线 | 例如算术指令、访存指令、特殊函数指令会使用不同资源 |
| 寄存器、共享内存等 | 保存线程状态和中间数据 | 存储资源也会影响一个 SM 能同时容纳多少线程块 |

不同 GPU 架构的 SM 布局、单元数量和指令吞吐不同，不应把某一张架构图当作所有 GPU 的固定结构。[课程相关段落](https://tvle9mq8jh.feishu.cn/docx/O9sTdvCGios7QTxVWZscgBamnWb)

## 3. Thread、Warp、Block 和 Grid

写 CUDA 核函数时，程序员组织的是线程：

- **Thread（线程）**：执行核函数的一个实例，有自己的索引和状态。
- **Block（线程块）**：一组线程，可通过块内共享内存协作，并使用块内同步。
- **Grid（线程网格）**：一次核函数启动所包含的线程块集合。

**Warp（线程束）由同一个 block 中的 32 个线程位置组成，是理解 GPU 指令执行的重要单位。** 最后一个 warp 可能不足 32 个有效线程；被条件分支屏蔽的线程也可能暂时不参与某条指令。[NVIDIA：Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)

例如，一个一维 block 有 128 个线程，就会划分为 4 个 warp：

```text
warp 0：threadIdx.x =   0～31
warp 1：threadIdx.x =  32～63
warp 2：threadIdx.x =  64～95
warp 3：threadIdx.x = 96～127
```

对于普通 CUDA 核函数，一个 block 的线程在同一个 SM 上执行；一个 SM 可以同时驻留多个 block，具体数量受寄存器、共享内存及硬件限制影响。**一个线程不固定绑定一个 CUDA Core，一个 block 也不独占一个 SM。**[CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/12.5.1/cuda-c-programming-guide/index.html)

一维线程的全局索引常写成：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

例如每块 128 个线程，`blockIdx.x = 2`、`threadIdx.x = 5`，则 `i = 261`。这里的索引是任务编号，不是 GPU 上的物理核心编号。[线程层级说明](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/02-basics/writing-cuda-kernels.html)

## 4. Warp Scheduler 如何隐藏延迟？

假设 warp A 发起一次访存，下一条指令需要等待返回的数据。如果 warp B 的指令已经就绪，调度器可以发射 B 的指令，让计算资源在等待期间继续工作。

这种方式称为**延迟隐藏**：等待没有消失，而是利用其他可执行任务覆盖部分等待时间。[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

但“线程越多一定越快”不成立。线程过多可能增加资源压力；如果所有 warp 都在等待，或者已经达到带宽上限，增加线程也不能消除瓶颈。性能需要结合实际工作负载测量。

## 5. 显存与片上存储：先分清位置和作用域

| 存储类型 | 典型位置 | 常规可见范围 | 用途 |
| --- | --- | --- | --- |
| Register（寄存器） | SM 内部 | 单个线程 | 保存临时变量和运算状态 |
| Shared Memory（共享内存） | SM 内部 | 同一个 block | 保存协作计算所需的中间数据 |
| Global Memory（全局内存） | 设备内存，通常由显存承载 | 可由不同 block 的线程访问 | 保存输入、输出及大数组 |
| Local Memory（局部内存） | 设备内存，并非 SM 内的专属高速存储 | 单个线程 | 承载部分线程私有数据，例如寄存器溢出 |

这里按普通 block 模型入门；更新架构中的线程块集群等扩展可以后续单独学习。L1/L2 cache 是硬件缓存概念，也不要直接等同于上述所有编程可见的存储空间。[设备内存空间说明](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/02-basics/writing-cuda-kernels.html)

## 6. 用自己的话重新表述

CUDA 让我们能够编写运行在 NVIDIA GPU 上的并行程序。GPU 的计算资源组织在多个 SM 中，每个 SM 包含运算、访存、调度和存储等资源。核函数的线程按 block 和 grid 组织，SM 中的调度器以 warp 为重要单位安排指令执行，让就绪的任务使用对应执行单元，并通过切换可执行任务隐藏部分等待延迟。


