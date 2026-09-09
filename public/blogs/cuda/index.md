# Cuda线程模型和显存模型

## 什么是cuda

CUDA 是 NVIDIA 提供的 GPU 通用计算平台和编程模型。CPU 更擅长：复杂控制逻辑和分支较多的程序，而GPU

更擅长：大量相似计算矩阵、向量、图像、张量处理。所以 CUDA 程序通常采用这样的分工：

```Plain Text
CPU：组织任务、分配内存、启动 Kernel、检查结果
GPU：执行大量并行计算
```

- 专为图形渲染而设计的**可编程图形处理器（GPU）**一开始是给打游戏使用的，这一转变背后的核心原因在于 GPU 架构的本质特征——它被专门优化用于**高密度、大规模并行计算任务**。随着AI时代的发展，AI 计算任务通常涉及大量的浮点运算，**其计算量大，数据频繁复用**的特点正好契合了 GPU 的设计优势。

- 与 CPU 不同，GPU 更倾向于将芯片上的晶体管资源集中用于**数据处理单元**，而不是像 CPU 那样大量投入于缓存管理和复杂控制逻辑的设计。这种**以计算为中心**的架构使其在执行大规模并行任务时表现出极高的效率。

GPU的结构是围绕一个流式多处理器（SM）的扩展阵列搭建的，NVIDIA GPU 中的 **SM（Streaming Multiprocessor，流式多处理器）** 是 GPU 的核心执行单元。每个 SM 包含多个功能单元（如 CUDA 核心、内存访问单元等），并负责协调这些单元之间的协作，以实现高效的并行计算。在后续内容中我们将看到，一个线程块（thread block）一旦被分配到某个 SM 上，就会在其上完成全部执行过程，不会迁移到其他 SM

一个 SM（Streaming Multiprocessor）通常由这些部分组成。不同 NVIDIA 架构的数量和名称会变化，但职责大致一致：

```Plain Text
SM
├── Warp Scheduler / Dispatch
├── CUDA Cores
│   ├── FP32/FP64 浮点单元
│   └── INT32 整数单元
├── Tensor Cores
├── Special Function Units（SFU）
├── Load/Store Units
├── Register File
├── Shared Memory / L1 Cache
└── 其他控制与数据通路
```

各部分作用：

- **Warp Scheduler（Warp 调度器）：**从多个 Warp 中选择能够执行的 Warp，并发射指令。

- **Dispatch Unit（分发单元）：把该 Warp 的下一条指令发给对应执行单元**

- **CUDA Cores：**执行普通的整数、浮点计算。一个 SM 内通常有很多 CUDA Core。

- **Tensor Cores：**专门加速矩阵乘法，深度学习中的 GEMM、Attention 等大量使用它。

- **SFU（特殊函数单元）：**计算 `sin`、`cos`、`exp`、倒数、平方根等特殊函数。

- **Load/Store Units：**负责从显存、Shared Memory 等位置读取和写入数据。

- **Register File（寄存器文件）：**每个线程使用的最快速存储空间。寄存器数量会限制一个 SM 同时驻留多少线程。

- **Shared Memory：**同一个 Block 内的线程共享的片上高速内存。它和 L1 Cache 通常共享一部分容量。

- **L1 Cache：**缓存线程经常访问的数据，减少访问更慢的显存。

执行过程大概是这样：

```Plain Text
线程
  ↓ 组成
Warp（通常 32 个线程）
  ↓ 由
Warp Scheduler 调度
  ↓ 使用
CUDA Core / Tensor Core / SFU / Load-Store Unit
  ↓ 访问
Register / Shared Memory / L1 / Global Memory
```

一个 SM 上通常同时驻留很多 Warp，但每个时钟周期不可能全部执行。调度器会检查：哪些 Warp 已经准备好。然后挑选一个或多个“就绪 Warp”。

比如我这个“warp schedule”选了某一个Warp，发现他的下一条指令是 `c = a + b;`

Dispatch Unit 就会把这条指令发给适合执行浮点加法的 CUDA Core；如果是矩阵乘法，就发给 Tensor Core；如果是访存指令，就发给 Load/Store Unit。

#### SM，Block，Warp之间的关系是啥

SM 是 **Streaming Multiprocessor（流式多处理器）**，可以理解为 GPU 内部真正负责执行 CUDA 线程块的基本硬件单元。一块 GPU 通常包含很多个 SM

```Plain Text
Grid  → 整个 GPU
Block → 一个或某个 SM
Warp  → SM 调度的基本单位
Thread → Warp 中的单个线程
```

```Plain Text
一个 SM
├── Block 0
│   ├── Warp 0：Thread 0 ~ 31
│   ├── Warp 1：Thread 32 ~ 63
│   └── ...
├── Block 1
│   ├── Warp 0
│   └── ...
└── Block 2 ...

```

- Block是程序员定义的线程集合,Block 会被整体分配到一个 SM

```Plain Text
kernel<<<gridDim, blockDim>>>(); // blockDim 决定每个 Block 中有多少线程。
```

- Warp 是硬件执行单位,NVIDIA 通常每 32 个线程组成一个 Warp。一个 Block 会被拆成若干个 Warp：

例如一个 Block 有 100 个线程：

```Plain Text
Warp 0：线程 0 ~ 31
Warp 1：线程 32 ~ 63
Warp 2：线程 64 ~ 95
Warp 3：线程 96 ~ 99
```

最后一个 Warp 中只有 4 个有效线程，其他 28 个位置闲置。

比如说

```Plain Text
gpu_hello<<<2, 3>>>();
代表
Grid
├── Block 0
│   └── 3 个线程，组成 1 个 Warp
└── Block 1
    └── 3 个线程，组成 1 个 Warp
```

虽然每个 Block 只有 3 个线程，但硬件仍按 32 线程的 Warp 组织，因此每个 Warp 实际只有 3 个有效线程。

#### GPU执行线程的时候的存储数据

Register：我自己的变量
Shared Memory：Block 内共享的变量
L1 Cache：当前 SM 自动缓存的数据
L2 Cache：整个 GPU 共享的缓存
显存：容量大，但访问慢

### cuda技术栈

在以上的背景下，NVIDIA顺势推出了一种GPU编程组件—— **CUDA（Compute Unified Device Architecture）** ，作为一种原生支持 GPU 编程的软硬件架构，CUDA 突破了过去必须通过图形 API 来使用 GPU 计算能力的限制，使得开发者可以直接在 GPU 上编写和执行通用计算程序。为了让大家对Cuda有一个直观理解，我们先来分层次看看它的结构

CUDA C/C\+\+ 只是在 C/C\+\+ 的基础上增加了 GPU 编程语法，例如：

1. `__global`\_\_：CPU 可以调用、GPU 执行的 Kernel

2. `__device`\_\_：只能在 GPU 上调用和执行

![image\.png](/blogs/cuda/19251c5aa4a9225b.png)

- **硬件驱动层（Cuda Driver）**：负责与 GPU 硬件通信，提供底层支持；

- **应用编程接口（CUDA Driver API）与运行时（CUDA Runtime API）**：为开发者提供简洁易用的编程接口和执行环境；

- **高级数学库**：

    - **CUBLAS**：用于线性代数运算的高性能库；

    - **CUFFT**：用于快速傅里叶变换的库。

同时这里有一个重要的组件在图中没有体现，那就是`nvcc`，也就是**NVIDIA CUDA 编译器（CUDA Compiler）**，它是 CUDA 工具链中的核心组件之一，专门用于编译含有 CUDA C/C\+\+ 扩展语法的源代码文件（通常以 `.cu` 为扩展名）。

- CUDA C/C\+\+ 还提供了丰富的运行时 API 和库函数（如 `cudaMalloc`、`cudaMemcpy`、`cudaLaunchKernel` 等），用于管理 GPU 内存、启动核函数以及实现 CPU 与 GPU 之间的数据传输与同步。

### 第一个程序

```C++
#include <cstdio>         *// 提供 printf()、fprintf()*
#include <cuda_runtime.h>
*// 体现了 CUDA 程序常见的分工：CPU 组织任务，GPU 执行并行计算*
*// __global__ 它表示这是一个 核函数（kernel）：这里由 CPU 发起调用，函数体在 GPU 上执行 *
__global__ void gpu_hello() {
    if (threadIdx.x == 1&& blockIdx.x == 1) {*  // 线程块编号为1线程块中的第0个线程执行下面的代码*
        printf("Hello from GPU!\n");
    }
    printf("线程块 %d，线程 %d\n", blockIdx.x, threadIdx.x);*   // threadIdx.x当前线程在线程块内的编号*
}

int main() {*               // 程序从 main() 开始，在 CPU 上执行 核函数名<<<线程块数量, 每个线程块的线程数量>>>(函数参数);*
    printf("CPU: Starting GPU task.\n");*    // cpu上面用 printf*
*    // 执行到 gpu_hello<<<1, 1>>>(); 时，CPU 就通知 GPU：“请执行这个函数。”*
    gpu_hello<<<2, 3>>>();* // 启动1个线程块 每个线程块一个线程** *

    *cudaError_t* error = cudaDeviceSynchronize();*  // 让 CPU 停在这里，等 GPU 完成任务后再继续。*
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    printf("CPU: GPU task finished.\n");
    return 0;
}

```

```Plain Text
nvcc hello.cu -o hello 
```

- CUDA Runtime/Driver 将 Kernel 启动请求提交给 GPU；GPU 硬件再将线程块分配到 SM，由 Warp Scheduler 按 Warp 调度指令，并由 Dispatch Unit 将指令发送到对应的执行单元

## 线程分层结构和管理（执行模型）

在并行计算中，线程是执行任务的最小实体。一个典型的线程包含以下组成部分：

- **程序：** 线程执行的代码逻辑。

- **程序计数器 \(PC\)：** 指向当前正在执行的指令地址。

- **上下文 \(Context\)：** 线程运行所需的环境信息。

- **内存与寄存器：** 线程存储临时数据和状态的地方。

在 CUDA 中，我们不是只运行一个或几个线程，而是利用 GPU 的硬件特性同时运行**成千上万个线程 \(Many threads\)**。

### 执行模型的分层

为了方便编程，CUDA 提供了层次化的线程组织方式。每个线程都通过一个三维向量 `threadIdx` 来标识，允许我们以一维、二维或三维的方式组织线程，构成一个线程块（Thread Block）。这种方式使得开发者可以更自然地对向量、矩阵或三维数据进行并行计算。

![image\.png](/blogs/cuda/b7e4838aa9963612.png)

- 图中的箭头可以理解为线程 —— 这是 CUDA 中最小的执行单元。多个线程可以组织成一个 **线程块（Thread Block）**，而多个线程块又组成一个 **网格（Grid）**。整个核函数所启动的所有线程都包含在这个网格之中。

- 与线程块内部的线程类似，网格层级也支持使用一维、二维或三维的方式来组织线程块，这为我们处理不同维度的数据结构提供了灵活性。需要注意的是，线程块中的线程数量是有限制的，因为同一个线程块内的所有线程必须运行在同一个流多处理器（Streaming Multiprocessor, SM）上，并共享该 SM 的有限资源（如寄存器、共享内存等）。

当启动一个核函数时，其网格中的线程块会被动态分配到 GPU 上各个可用的 SM 中执行。一旦某个线程块被调度到某个 SM 上，其中的所有线程将始终在该 SM 上并发执行，不会迁移到其他 SM 上。由于 GPU 包含多个 SM，多个线程块可以在不同的 SM 上同时运行，实现真正的物理级并行，而不是像 CPU 那样依赖时间片轮转。SM充当了硬件级的实时控制中枢，在硬件资源允许范围内驱动大量线程并发执行，最终依靠内部的束调度器（Warp Scheduler）精准调度指令，确保海量计算任务高效、有序地在核心上推进。

用一句简单的话来说就是，当一个block被分配给一个SM后，他就只能在这个SM上执行了，不可能重新分配到其他SM上了，多个线程块可以被分配到同一个SM上。此外，线程块中的线程会以 **32 个线程为一组** 进行调度（调度的时候受限一个SM中的可用资源），这组线程被称为 **线程束（Warp）**。每个线程束中的线程会**同时执行相同的指令**，但各自使用不同的数据。每个线程拥有独立的程序计数器和寄存器状态，以实现数据级别的并行。SM 将线程块划分为多个 warp，并根据硬件资源调度它们执行，Warp Scheduler 将选中的 warp 的指令发送给对应的 Dispatch Unit（调度单元），随后Dispatch Unit 再将该指令分发到相应的功能单元：CUDA Cores、LD/ST 或 SFU。

当一个 Warp 因为等待内存读取（长延迟操作）而阻塞时，SM 的硬件调度器会**瞬间**切换到另一个处于就绪状态的 Warp 执行。如果一个 Warp 的下一条指令所需的数据（操作数）已经准备好了，它就处于“就绪”状态，可以被执行。

- 注意一个 Block 被分配到 SM 后，会被拆分成多个 Warp。此时 Warp 已经“驻留”在 SM 上，但不一定马上能执行。只有满足条件时，Warp 才是可调度的，例如：下一条指令的操作数已经准备好，没有等待前一条指令的结果，这个时候再有SM 中的 Warp Scheduler 会不断寻找可调度 Warp：

```Plain Text
Warp A：等待显存，不可调度
Warp B：操作数就绪，可调度
Warp C：等待同步，不可调度
Warp Scheduler → 选择 Warp B
```

然后把 Warp B 的下一条指令交给 Dispatch Unit，再发送到 CUDA Core、LD/ST 或其他执行单元。需要哪个数据就用LD/ST作加载，需要计算单元就用其他单元

![image\.png](/blogs/cuda/c74a9058a9ba362a.png)

![image\.png](/blogs/cuda/4448185bda24b3fe.png)

从图中可以清晰地看到我们前面介绍的三级线程分层结构：**线程（Thread）→ 线程块（Block）→ 网格（Grid）**。图中虽然没有明确画出 **warp** 的层级，但你可以这样理解：**warp 是在线程块内部，以 32 个线程为一组进行划分的基本调度单元**。

一个 Block 被分配到 SM 后，会被拆分成多个 Warp。此时 Warp 已经“驻留”在 SM 上，但不一定马上能执行。只有满足条件时，Warp 才是可调度的，例如：下一条指令的操作数已经准备好，没有等待前一条指令的结果，这个时候再有SM 中的 Warp Scheduler 会不断寻找可调度 Warp。然后把 某个 Warp  的下一条指令交给 Dispatch Unit，再发送到 CUDA Core、LD/ST 或其他执行单元。需要哪个数据就用LD/ST作加载，需要计算单元就用其他单元



