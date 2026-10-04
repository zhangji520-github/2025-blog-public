# softmax算子的cuda实现

> 本节将介绍Softmax算子的CPU和GPU实现，重点讲解了如何通过共享内存、线程块归约及warp洗牌指令优化CUDA实现，以提升并行计算效率与性能。

## Softmax算子

### 简介

- 在分类任务中，模型通常输出一个实数向量，表示每个类别的原始得分（`logits`）。这些得分尚未经过归一化处理，因此不能直接解释为类别概率。为了将这些得分转化为具有概率意义的分布，通常使用**Softmax**函数进行处理。**Softmax**的作用是将输入向量映射到一个所有元素值域在`[0, 1]`之间、且总和为 1 的概率分布上，从而使得我们可以将其解释为各个类别的预测概率。
- 正因这一特性，**Softmax**被广泛应用于多分类问题中，如图像分类、文本分类等任务，同时也是Transformer架构中注意力机制的重要组成部分。

### 计算公式

给定一个输入向量 $\mathbf{z} = [z_1, z_2, ..., z_n] \in \mathbb{R}^n$，Softmax的原始公式定义如下：$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}, \quad \forall i = 1, 2, ..., n$

- $z_i$是第$i$个类别的原始得分，也就是上文中说的`logits`
- 分母是对所有类别得分`logits`的指数求和，起到归一化的作用

但是直接计算**Softmax**可能会出现数值不稳定的问题，尤其是当某些$z_i$很大时，$e^{z_i}$可能溢出，导致`NaN`或`Inf`。为了避免这个问题，通常会对输入进行平移操作，也就是在分子中减去$z$的最大值，这样一来就能保证最大的指数项为 0，避免了指数爆炸问题。$\text{Softmax}(z_i) = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_{j=1}^{n} e^{z_j - \max(\mathbf{z})}}$

### 实现方式

1. **计算最大值：**根据以上的公式，我们可以写出的计算的流程，由于采用的是加入了平移技巧的计算方式，因此首先需要计算输入$z$中的最大值，也就是$max(z)$，$m = max(z)$，其中$z = [z_1, z_2, \dots, z_n]$是**Softmax**的输入，**$m$是输入的最大值。**
2. **求分母：**有了本次输入中的最大值m，我们就可以求$\sum_{j=1}^{n}e^{z_j-m}$，其中的$m$是我们在步骤一中求出的最大值。该公式表示将每个输入值减去最大值 $m$ 后，再进行指数运算并求和。
3. **求出结果：**有了前两步中求出的最大值$m$和分母$\sum_{j=1}^{n} e^{z_j - m}$，我们就可以对数量为$n$的输入$z$进行循环遍历，依次计算每个元素的 Softmax值，从而得到最终的结果。

## CPU实现

有了上述的叙述，下面我们先给出一个CPU实现版本。其主要目的是帮助大家进一步熟悉该公式的计算流程，同时也为后续与 CUDA 版本的代码进行对比做准备，确保两者在执行结果上保持在一定的误差范围之内。完整代码见**course3/softmax_naive.cu**。

1. 首先我们来看一下这个函数的传参部分，`softmax_forward_cpu`的输入是一个N×C维度的张量，记作`inp`；

```C++
void softmax_forward_cpu(float *out, const float *inp, int N, int C) {
  for (int i = 0; i < N; i++) {
    const float *inp_row = inp + i * C;
    float *out_row = out + i * C;
```

1. 对于输入的 N 行数据（N个输入向量），我们将逐行进行处理。其中变量 `i` 表示当前正在处理的行号，`inp_row` 指向输入张量中第`i`行的起始位置，`out_row` 同理，指向输出张量中对应行的起始位置；
2. 随后我们将计算出时每行输入（循环中是第i行）中的最大值`maxval`，也就是公式中的$m$；因为我们输入的维度是N × C，所以现在对长度为C的一行进行遍历找出其中的最大值。

```C++
for (int i = 0; i < N; i++) {
    ...
    ...
    float maxval = -INFINITY;
    for (int j = 0; j < C; j++) {
      if (inp_row[j] > maxval) {
        maxval = inp_row[j];
      }
    }
```

1. 计算出公式中对应的分母，也就是上一节说的需要将**每个输入值减去最大值 $m$ 后，再进行指数运算并求和，**我们记作`sum`**。**

```C++
for (int i = 0; i < N; i++) {
    ...
    ...
    float sum = 0.f;
    for (int j = 0; j < C; j++) {
      out_row[j] = expf(inp_row[j] - maxval);
      sum += out_row[j];
    }
```

1. 有了分母和最大值，我们将分母记作`norm`，随后就可以对**一行中数量为$C$的输入$z$进行循环遍历**，依次计算每个元素的Softmax值，从而得到最终的结果`out_row`。

```C++
for (int i = 0; i < N; i++) {
    ...
    ...
    float sum = 0.f;
    for (int j = 0; j < C; j++) {
      out_row[j] = expf(inp_row[j] - maxval);
      sum += out_row[j];
    }
    
    float norm = 1.f / (float)sum;
    for (int j = 0; j < C; j++) {
      out_row[j] *= norm;
    }
```

至此，我们已经完成了在 CPU 上的 Softmax 算子实现。为了便于后续对比，我们暂时不运行该代码，而是留到与在 CUDA 上实现的计算结果进行对比时再统一执行和验证。

## GPU实现

### Cuda基础实现

1. 回顾第二章中介绍的CUDA执行模型，我们知道 GPU 上的线程是以**Block（线程块）**和 **Thread（线程）**的方式进行组织的。一个线程块中可以包含多个线程，多个线程块则组成了一个 Grid（网格）。
2. Softmax的输入是一个形状为N×C的张量，**其中N表示行数，C 表示每行中的元素个数**。基于这一结构，我们初步的设计思路是：为每一行分配一个独立的线程块（即总共配置 N 个线程块），该线程块内的所有线程共同完成当前行的归约操作（包括最大值的计算和指数和的累加）。这样可以充分利用 GPU 的并行性，对每一行数据进行高效的 Softmax 计算。一个线程块处理的数据量就是C，总共有N个线程块。

#### 实现方式

我们首先实现一个基础版本的 CUDA Softmax 核函数。该版本为每一行输入数据分配一个线程块，总共配置 `N` 个线程块（对应矩阵的 `N` 行），但每个线程块中仅包含 **一个线程**（即 `blockDim.x = 1`）。这个唯一的线程负责完成对应行上所有元素的归约操作，包括求取最大值（用于数值稳定性）和指数加和。

由于每个线程块只使用单个线程，并且该线程串行处理整行数据，因此该实现**并未有效利用 GPU 的大规模并行计算能力**。从性能角度看，这是非常低效的。然而，这种设计在功能层面是正确的，可作为后续优化的基础参考版本。

```C++
// CUDA kernel
__global__ void softmax_forward_kernel1(float *out, const float *inp, int N,
                                        int C) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // threadIdx.x 等于[0，N) i的取值是[0，N)
  if (i < N) {
    const float *inp_row = inp + i * C;
    float *out_row = out + i * C;

    float maxval = -INFINITY;
    for (int j = 0; j < C; j++) {
      if (inp_row[j] > maxval) {
        maxval = inp_row[j];
      }
    }
    float sum = 0.f;
    for (int j = 0; j < C; j++) { // 只有一个线程做遍历
      out_row[j] = expf(inp_row[j] - maxval);
      sum += out_row[j];
    }
    for (int j = 0; j < C; j++) {
      out_row[j] /= (float)sum;
    }
  }
}
```

#### 启动函数和性能对比

根据以上分析，我们在调用核函数时将配置`N`个线程块，每个线程块中仅有一个线程实际参与计算，负责完成对应行的完整归约操作。基于这一设计，我们得到了如下的核函数启动方式。如果需要运行这里的代码，需要把course3/softmax_naive.cu第294行，修改为如下的调用。

```C++
int blockSize = N;
int numBlocks = 1;
softmax_forward_kernel1<<<numBlocks, blockSize>>>(d_out, d_inp, N, C);
```

在 `course3/softmax_naive.cu` 中，我们完整实现了该核函数，并在 CPU 上对应实现了相同的 Softmax 运算以进行结果验证。最终的运行结果显示，两者的计算结果是一致的。但是，在输入维度为 32×4096 的测试场景下，这种最普通的Cuda实现方式相比 CPU 并没有优势。

```Bash
Results match: YES
CPU time: 13.6884 ms
GPU time: 4.72269 ms
Speedup: 2.89843x
```

#### 使用nsight-compute进行分析

具体分析文件见ncu文件夹course3/softmax_1.ncu-rep

当前实现存在明显的"双低"问题：SM利用率和显存利用率均偏低，即Compute Throughput和Memory Throughput两个关键指标表现不佳。这表明计算单元和内存带宽都未得到充分利用，根本原因在于并行度不足，无法有效发挥硬件性能。

### 优化1：用共享显存和块内归约加速

#### 实现方式

在之前的实现中，**我们为每一行数据仅分配了一个线程进行处理，其中一个线程串行处理一行，这导致了在输入张量第二维度上的并行度严重不足。**为了提升该维度的计算效率，我们考虑在这一维度上引入更多的线程以增强并行性。

然而，如何在一个线程块内部协调多个线程，使其高效地协作完成一行数据中的最大值和总和的计算，是一个值得深入思考的问题。我们的解决方案如下：假设输入张量在第二维度上的大小为 512，我们决定使用一个包含 16 个线程的线程块来协同处理这一行数据。具体思路分为两个阶段：

1. **局部计算阶段**  
每个线程首先独立处理一部分数据，计算其负责范围内的局部最大值与局部和。由于总共有 512 个元素和 16 个线程，每个线程将负责 32 个数据点（即$512 / 16 = 32$）。例如，线程 0 处理索引为 0、32、64、96、… 的元素，线程 1 处理索引为 1、33、65、97、… 的元素，以此类推。通过这种方式，每个线程在其所负责的数据范围内独立计算出一个局部最大值和一个局部和。最终，我们共获得 16 个局部最大值和 16 个局部和。
2. **归约阶段**  
**在第一阶段结束后，我们需要对每个线程的局部结果进行归约操作，以得到全局最大值和全局和。**对于最大值，我们需要在这 16 个局部最大值中找出最大值；而对于总和，则需要将 16 个局部和累加起来，得到最终的全局和。这个归约过程可以在同一个线程块内部通过共享内存和同步机制高效地完成。
3. 这样就用一个线程块得到了一行当中的最大值，而最大值是计算`softmax`必须的。

通过上述策略，我们不仅提升了第二维度上的并行度，还有效利用了线程块内线程之间的协作能力，从而显著提高了整体计算效率。我们来看`course3/softmax_naive.cu`中的相关实现：

1. 我们仍然采用一个线程块来处理一行数据。首先，需要定位到当前所处理的行，即获取指向该行起始位置的指针 `float* x`。
2. 同时，**为了存储每个线程在第一阶段计算得到的局部最大值，我们使用共享显存（shared memory）中的数组进行记录。**这样可以确保同一个线程块内的所有线程都能快速访问这些中间结果，为后续的归约操作做好准备；

   ```C++
   __global__ void softmax_forward_kernel2(float *out, const float *inp, int N,
                                           int C) {
     extern __shared__ float shared[];
     int idx = blockIdx.x;   
     int tid = threadIdx.x; 
     int block_size = blockDim.x;
     const float *x = inp + idx * C;  
   ```
3. **求线程的局部最大值**

   - 接下来，我们让同一个线程块中的每个线程独立计算其负责数据范围内的局部最大值，并将结果写入共享内存 `shared` 中对应的位置。
   - 如第2段所述，每个线程（线程ID为 `tid`）依次处理输入数据中下标为 `tid`、`tid + block_size`、`tid + 2 × block_size` 的元素，依此类推，直到遍历完其所负责的所有数据，也就是上文中对线程0的0，32，64。**最终，每个线程将其计算得到的局部最大值存储在 `shared[tid]` 中，供后续归约阶段使用，**对应第5行。

   ```C++
   float maxval = -INFINITY;
   for (int i = tid; i < C; i += block_size) {
     maxval = fmaxf(maxval, x[i]);
   }
   shared[tid] = maxval;
   __syncthreads();
   // reductions
   for (int stride = block_size / 2; stride >= 1; stride /= 2) {
     __syncthreads();
     if (tid < stride) {
       shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
     }
   }
   ```
4. **求线程间的全局最大值**

   - 在所有线程完成各自负责数据的局部最大值计算并将其写入共享内存`shared`后，下一步是通过归约操作求取这些局部最大值中的全局最大值。我们采用一种层次化的并行比较策略：以16个线程为例，在第一轮比较中，线程0与线程8进行比较，将二者中的较大值写回 `shared[0]`；同理，线程1与线程9比较，结果存入`shared[1]`，依此类推。
   - 在下一轮中，比较的跨度减半：线程0与线程4进行比较，并将最大值写入 `shared[0]`，线程1与线程5比较，结果写入 `shared[1]`，以此类推。**如此反复，每次比较的跨度为上一轮的一半，直到最终在 `shared[0]` 中得到整个数组的最大值。**

   ```C++
   __syncthreads();
   float offset = shared[0];
   ```

```C++
  // block_size = 16, stride = 8，以16个线程为例
  // tid取值为[0,16)，在第一轮循环中只有tid[0,8)工作
  for (int stride = block_size / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (tid < stride) {
      // 第一轮
      // tid = 0时 0 + 8
      // tid = 1时 1+9
      // tid = 2时 2+10
      // 第二轮
      // stride = 4 在第二轮循环中只有tid[0,4)工作
      // tid = 0时 0 + 4
      // tid = 1时 1 + 5
      // tid = 2时 2 + 6
      shared[tid] += shared[tid + stride];
    }
  }
```

我们使用16个线程进行归约操作，因此整个过程共需执行4轮循环（即$\log_2(16) = 4$），每一轮中比较的线程对之间的跨度依次减半。最终，在最后一轮结束后，`shared[0]`中将保存所有局部最大值中的全局最大值。这一分层归约策略既可用于求和，也可用于求最大值，仅操作类型不同。通过这种方式，我们能够高效地在线程块内部完成最大值或总和的归约计算。

```C++
float maxval = -INFINITY;
for (int i = tid; i < C; i += block_size) {
   maxval = fmaxf(maxval, x[i]);
}
```

C = 4096，线程块配置的比较小，比如block_size = 128，每个线程就要处理32个数据。在这里，对于任意一个线程，我都要求出它负责的32个数据的当中的最大值。

**最后，让我们来看一遍完整的代码：**

```C++
__global__ void softmax_forward_kernel2(float *out, const float *inp, int N, int C) {
  extern __shared__ float shared[];
  int idx = blockIdx.x;   // ranges [0, N)
  int tid = threadIdx.x;  // ranges [0, block_size)
  int block_size = blockDim.x;
  const float *x = inp + idx * C;  // idx-th row of inp
  // thread coarsening
  float maxval = -INFINITY;
  for (int i = tid; i < C; i += block_size) {
    maxval = fmaxf(maxval, x[i]);
  }
  shared[tid] = maxval;
  __syncthreads();
  // reductions
  for (int stride = block_size / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
  }
  __syncthreads();
  float offset = shared[0];
  // compute expf and write the result to global memory
  for (int i = tid; i < C; i += block_size) {
    out[idx * C + i] = expf(x[i] - offset);
  }
  __syncthreads();
  // thread coarsening again, for the sum
  x = out + idx * C;  // idx-th row of out
  float sumval = 0.0f;
  for (int i = tid; i < C; i += block_size) {
    sumval += x[i];
  }
  shared[tid] = sumval;
  __syncthreads();
  // reductions
  for (int stride = block_size / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
  }
  // broadcast the sum to all threads in the block
  __syncthreads();
  float sum = shared[0];
  // divide the input values by the sum
  for (int i = tid; i < C; i += block_size) {
    out[idx * C + i] = x[i] / sum;
  }
}
```

#### 启动函数和性能对比

最终的运行结果显示，两者的计算结果是一致的，在输入维度为 320× 4096 的测试场景下，**与第一种方式比有了非常明显的提升。**GPU上的执行时间从4.72ms降到了0.0399ms。另外，它的启动方式是和基础实现保持一致的。

```C++
int blockSize = 128; //每个线程处理32个数据
int numBlocks = N;
softmax_forward_kernel2<<<numBlocks, blockSize>>>(d_out, d_inp, N, C);
```

```Plain Text
Results match: YES
CPU time: 13.9473 ms
GPU time: 0.039936 ms
Speedup: 349.241x
```

#### 使用nsight-compute进行分析

具体分析文件见ncu文件夹course3/softmax_2.ncu-rep

总体来看，计算单元和访存部件的利用率虽有非常明显的提升，但从整体指标来看提升幅度仍然有限。这主要是因为我们传入的计算数据量过少，无法充分发挥GPU的并行处理能力。接下来我们将重点分析造成当前kernel性能瓶颈的根本原因。

在每个调度器最多支持12个warp的情况下，此工作负载平均每个调度器仅有3.45个活跃warp，而每周期平均只有0.41个warp具备执行条件。这表明大部分warp都因各种原因被阻塞，无法及时获得执行机会。接下来我们将深入分析造成这种阻塞的具体原因。

通过分析可以找到问题的根本原因。提示信息显示：平均而言，此工作负载中的每个warp花费4.2个周期处于停顿状态，等待L1TEX（本地、全局、表面、纹理）操作的记分牌依赖关系。进一步分析发现，导致warp等待时间最长的原因主要有两个：一是在Block进行归约操作时使用`__sync_threads()`导致的warp等待和同步；但最主要的原因是Stall Long ScoreBoard，根据参考资料，这表示等待全局显存数据就绪的停顿。

在我们的代码中，最长的warp停顿发生在第73行。从寄存器依赖关系分析，导致第73行stall的原因是对R4寄存器的加载操作，该数据需要从全局显存中读取，从而造成了显著的等待延迟。**换句话说延迟主要还是由于加载全局显存引起的。**

**这里参考了文章：**

1. https://zhuanlan.zhihu.com/p/604099796
2. https://docs.nvidia.com/nsight-compute/ProfilingGuide/

## 线程束洗牌指令

### 线程束（Warp）

在第2节课程中，我们已经学习了以下关于 CUDA 执行模型的基本知识：

- 一个线程块（thread block）会被硬件划分为多个线程束（warp）；
- 在现代 CUDA 架构中，每个 warp 由**32 个线程**组成，一个warp中的32个线程并发执行相同的指令。

### 束内线程（Lane）

每个 warp 内部包含 32 个线程，这些线程被称为**束内线程（lane）**：

- 每个线程束内的线程有一个唯一的索引，称为 lane index（束内线程索引），其取值范围为 `[0, 31]`；
- 我们可以通过如下方式计算当前线程的全局索引：

  ```C++
  int globalThreadId = threadIdx.x +
                       threadIdx.y * blockDim.x +
                       threadIdx.z * blockDim.x * blockDim.y;
  ```
- 进一步，可以通过以下公式确定该线程所属的 warp 编号及其在 warp 中的 lane 编号。例如，线程块中的线程 1 和线程 33 具有相同的束内线程索引（lane ID = 1），但由于它们位于不同的线程束中，因此具有不同的 warp ID，分别是（0和1）。

  ```C++
  int warpId = globalThreadId / 32;
  int laneId = globalThreadId % 32;
  // int blockSize = 128; 128个线程，可以分为4个warp，每个warp当中有32个线程
  ```

而线程束洗牌函数是一类作用于 **线程束（warp）内部** 的协同操作函数，其主要目的是允许同一个 warp 中的不同线程直接访问彼此的寄存器数据，从而提供一种高效的信息交换机制。**与优化1中的kernel通过共享内存或全局内存进行线程间通信的方式不同，洗牌操作通过寄存器实现数据交换，具有以下优势：**

- 数据交互延迟极低；
- 不消耗额外内存带宽；
- 显著提升 warp 内部数据通信效率。

所以在需要频繁进行束内线程协作的场景下（如归约、广播、扫描等），使用洗牌函数可以带来显著的性能提升。

### 常见的洗牌指令

```C++
// 从特定的束内线程获取数值，跨线程束值的广播
T __shfl_sync(unsigned int mask, T var, int srcLane, int width = warpSize);
// 通过线程束上移获取数值
T __shfl_up_sync(unsigned int mask, T var, unsigned int delta, int width = warpSize);
// 通过线程束下移获取数值
T __shfl_down_sync(unsigned int mask, T var, unsigned int delta, int width = warpSize);
```

#### \_\_shfl_sync

所有参与操作的线程从指定的源线程（由`srcLane`指定）中读取变量`val`的值，并将该值广播到所有参与线程（包括源线程自身）。换句话说，该操作实现了 warp 内部的**单点广播（broadcast from a single lane）**，即将某个特定 lane 中的数据快速分发给 warp 中的其他线程。

当这里的`srcLane`等于2时，线程2中的`val`变量会被一个warp内的其他线程接收。为了让大家更直观的理解，我们写了以下的代码`course3/shuffle.cu`来演示`__shfl_sync`函数的功能。

```C++
__global__ void test_shuf_broadcast(int *dOutput, const int *dInput,
                                    const int srcLane) {
  int val = dInput[threadIdx.x];
  val = __shfl_sync(0xFFFFFFFF, val, srcLane, 32);
  dOutput[threadIdx.x] = val;
}

int main() {
  const int numThreads = 32;
  const int srcLane = 2;

  int hInput[numThreads];
  int hOutput[numThreads];

  for (int i = 0; i < numThreads; ++i) {
    hInput[i] = i;
  }

  int *dInput, *dOutput;
  cudaMalloc(&dInput, numThreads * sizeof(int));
  cudaMalloc(&dOutput, numThreads * sizeof(int));

  cudaMemcpy(dInput, hInput, numThreads * sizeof(int), cudaMemcpyHostToDevice);
  ...
  ...
}
```

我们的每个线程中都定义了`val`变量，随后通过`__shfl_sync`函数将线程2中的`val`值赋值给其他线程。结合实际运行结果可以验证上述分析：从输出结果中可以看出，所有有效线程中的`val`值均已变为 2，也就是所有线程会从val = \_\_shfl_sync(0xFFFFFFFF, val, srcLane, 32) ，就是dInput[2]中获取数据。

```Bash
test_fss@node4:~/code/cuda_code/build/course3$ ./shuffle 
Broadcasting value from thread 2:
hOutput[0] = 2
hOutput[1] = 2
hOutput[2] = 2
hOutput[3] = 2
hOutput[4] = 2
hOutput[5] = 2
hOutput[6] = 2
hOutput[7] = 2
...
```

#### \_\_shfl_up_sync

在一个warp中，lane id为`t`的线程将从lane id为`t - delta` 的线程中读取变量 `val` 的值；若`t - delta < 0`，则该线程保留自身的原始值`val`。这是一种数据向上平移的过程。

<sheet sheet-id="DQZ2ho" token="WsqcsmnMchdDqRtweMocUTg6nzS"></sheet>

这种操作实现了 warp 内数据的**向上平移**效果：线程0和1的值保持不变，而线程2及其之后的线程依次**“获取”**之前相隔 `delta` 的线程中的值。

我们将用一个示例来演示`__shfl_up_sync`函数的作用，完整代码见`course3/shuffle_up.cu`，在这里线程号2之后的线程将从之前的线程中获取对应的`val`值。

```C++
__global__ void test_shuf_up_sync(int *dOutput, const int *dInput) {
  int val = dInput[threadIdx.x];
  val = __shfl_up_sync(0xFFFFFFFF, val, 2, 32);
  dOutput[threadIdx.x] = val;
}
```

编译并执行

```Bash
hOutput[0] = 0
hOutput[1] = 1
hOutput[2] = 0
hOutput[3] = 1
hOutput[4] = 2
hOutput[5] = 3
hOutput[6] = 4
hOutput[7] = 5
hOutput[8] = 6
hOutput[9] = 7
hOutput[10] = 8
```

#### \_\_shfl_down_sync

该函数是 `__shfl_up_sync` 的反向操作，用于在 warp 内实现**向下偏移（downward shift）**的数据交换。换句话说，lane ID 为 `t` 的线程将从 lane ID 为 `t + delta` 的线程中读取变量 `val` 的值；若 `t + delta >= width`，则保留当前线程自身的原始值 `val`。

当delta为2时，thread 0从thread 2中获取对应的值，thread 1从thread 3中获取对应的值，直到线程 t，t + delta > 32，那么该线程就直接保留线程自身的值。我们将用一个示例来演示`__shfl_down_sync`函数的作用，完整代码见`course3/shuffle_down.cu`，在这里线程号0之后的线程，将从它之后的线程中获取对应的`val`值，直到t + delta > 32，此处delta等于2，所以最后的两号线程 t = 30, 31会获取其自身的值。

```Bash
...
hOutput[26] = 28
hOutput[27] = 29
hOutput[28] = 30
hOutput[29] = 31
hOutput[30] = 30
hOutput[31] = 31
```

### 优化2：用warp洗牌指令加速

学了上面的warp内通信指令，现在通过显式 warp 级编程获得更高的性能。并行程序通常使用集体通信操作，例如并行缩减和扫描。在此处则是使用`__shfl_down_sync` 函数进行操作，**需要注意的是此时必须要`block_size=32`，也就是一个线程块中刚好只有一个warp。**

在掌握了 warp 洗牌操作的基本原理之后，我们再来理解如何通过洗牌指令实现高效的**线程束内归约（warp-level reduction）**就变得清晰明了。假设我们当前需要对一个 warp 中的8 个元素进行求和操作。为了高效完成这一任务，我们可以采用多轮洗牌加法的方式逐步归约，最终在某个线程中得到总和结果。

#### 归约流程

##### 第一轮归约：delta = 4

我们在第一轮中调用如下函数：

```C++
v += __shfl_down_sync(0xFFFFFFFF, v, 4);
```

- `mask = 0xFFFFFFFF`：表示 warp 中所有线程都参与本次操作；
- `delta = 4`：每个线程从其 lane ID 向后偏移 4 的位置读取值；
- 执行后：

  - 线程 0 获取线程 4 的值并与自身相加；
  - 线程 1 获取线程 5 的值并与自身相加；
  - 线程 2 获取线程 6 的值并与自身相加；
  - 线程 3 获取线程 7 的值并与自身相加；
  - 线程 4\~7 的值也保持不变（因为它们超出下一轮有效范围，但仍会执行该语句）；

此时，线程 0\~3 中分别保存了两组两个元素的局部和。

##### 第二轮归约：delta = 2

接下来我们将偏移量减半，继续执行：

```C++
v += __shfl_down_sync(0xFFFFFFFF, v, 2);
```

此时，线程 0 获取线程 2 的值并与自身相加，线程 1 获取线程 3 的值并与自身相加。最终，线程 0 中保存的是前四个元素的总和，而线程 1 中保存的是后四个元素的总和。

##### 第三轮归约：delta = 1

最后一步将偏移量设为 1：

```C++
v += __shfl_down_sync(0xFFFFFFFF, v, 1);
```

线程 0 获取线程 1 的值并与自身相加。最终，线程 0 中保存的就是整个 8 个元素的总和。

#### 实现最大值计算

在掌握了上述背景知识之后，我们便可以编写用于计算输入数据中最大值与总和的核函数。以下是我们的具体实现方案，完整代码请参考 `course3/softmax_naive.cu`。

在第一轮归约（reduction）过程中，一个warp中的 32 个线程协同工作：

1. 例如，线程 0 会将其局部变量 `val` 与线程 16 的 `val` 值相加，线程 1 会与线程 17 的 `val` 相加，依此类推。
2. 这一阶段的目标是逐步将多个线程中的局部结果合并，最终在一个线程中得到全局的最大值或总和。

```C++
__device__ float warpReduceMax(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
  }
  return val;
}
```

#### 完整的计算流程

1. 在本次归约操作中，我们通过配置一个线程块来启动核函数。该线程块包含 32 个线程，每个线程负责输入数据中的特定位置。例如，在处理 128 个元素的情况下，线程 0 负责索引为 0、32、64 和 96 的元素，线程 1 负责索引为 1、33、65 和 97 的元素，以此类推。
2. 通过这种方式，每个线程计算其负责部分的局部和（或局部最大值 max_val）。**随后，这些局部结果将通过 warp 内的洗牌指令（shuffle instructions）在32个线程之间进行交换与合并，逐步累加或比较以获得最终的全局最大值。**
3. 这一过程的具体实现流程与上一节所述的归约策略一致，最终使我们能够高效地在单个线程块内完成对输入数据的最大值计算。

```C++
__global__ void softmax_forward_kernel3(float *out, const float *inp, int N,
                                        int C) {
  extern __shared__ float shared[];
  int idx = blockIdx.x;
  int tid = threadIdx.x;
  const float *x = inp + idx * C;

  float maxval = -INFINITY;
  for (int i = tid; i < C; i += blockDim.x) {
    maxval = fmaxf(maxval, x[i]);
  }
  maxval = warpReduceMax(maxval);

  ...
  ...
}
```

#### 启动函数和性能对比

在启动核函数之前，需要注意一点：**在这种实现方式中，线程块中的线程数量只能设置为 32，即一个 warp 的大小**，以确保能够正确使用 warp 内的洗牌指令（shuffle instructions）进行数据交换与归约操作。

```C++
int blockSize = 32;
int numBlocks = N;
softmax_forward_kernel3<<<numBlocks, 32>>>(d_out, d_inp, N, C);
```

可以看到，由于在一个线程块中只使用了一个warp大小的线程进行归约计算，这里的加速效果并没有优化1中那么明显。

```Plain Text
Results match: YES
CPU time: 13.4931 ms
GPU time: 0.147456 ms
Speedup: 91.5057x
```

#### 使用nsight-compute进行分析

具体分析文件见ncu文件夹course3/softmax_3.ncu-rep

从整体来看，这个kernel的计算效率和访存效率都远低于第二种优化方案。作者认为主要原因是每个线程块的规模较小，并行度不足。具体而言，当前方案中每32个线程需要维护1024个数据的归约操作，而之前的方案是128个线程处理同等规模的数据归约。显然，后者对硬件资源的利用率更高。从活跃/可用warp的角度来看，相比上一种优化方案也明显更少，这更容易理解：本来可用的warp数量就有限，一旦遇到资源阻塞，能够有效执行的warp就更少了。

```C++
  int blockSize = 32;
  int numBlocks = N;
  for (int i = 0; i < warm_up; ++i) {
    softmax_forward_kernel3<<<numBlocks, blockSize>>>(d_out, d_inp, N, C);
  }
```

### 优化3：用warp洗牌指令和共享内存进行加速

#### 实现

我们注意到，**优化2**存在一个明显的局限性：它要求线程块大小必须等于 32（即一个 warp 的大小），这在一定程度上限制了线程块内部更大的并行潜力。为了解决这一问题，我们引入共享内存（shared memory），以支持更大线程块中多个 warp 的协同计算，从而提升整体并行效率与计算吞吐量。

新版本的代码结构与之前类似，但由于引入了更多的 warp 并行，因此需要借助共享内存来存储各个 warp 的局部结果。具体来说：

1. 如前所述，每个 warp 都需要一个位置来存储其局部最大值和局部和。因此，我们在共享内存中分配大小为 `warpsPerBlock × 2`的数组，其中前 `warpsPerBlock` 个位置用于存储每个 warp 的局部最大值，后 `warpsPerBlock` 个位置用于存储对应的局部和；
2. 在代码的第 16–19 行中，每个线程仍然负责计算其所属数据段的局部最大值。随后，通过 warp 内部的归约操作（如 shuffle 指令），**最终在一个 warp 内得到该 warp 所处理数据的最大值，**也就是第19行中的`maxval`；

4个warp来执行，blockDim.x = 128，一个warp总共有32个线程，总共4个warp 128 threads，一行有4096个数据，所以一个线程处理32个数据。

**第一步：**

1. warp1的最大值，0-31号线程的最大值，32×32 = 1024。32表示32个线程，每个线程处理32个数据，所以一个warp处理1024个数据
2. warp2的最大值，32-63号线程的最大值，32×32 = 1024
3. warp3的最大值，64-95号线程的最大值，32×32 = 1024
4. warp4的最大值，95-128号线程的最大值，32×32 = 1024

**第二步：**

[warp1最大值，warp2最大值，warp3最大值，warp4最大值] 放到共享显存当中，得到4个warp当中的最大值，也就是4096个数据的最大值，也就是一行的最大值。

```C++
__global__ void softmax_forward_kernel4(float* out, const float* inp, int N, int C) {
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; 
    int laneId = threadIdx.x % 32;

    int warpsPerBlock = blockDim.x / 32;

    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    const float* x = inp + idx * C;

    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, x[i]); //一个线程得到32个数据的最大值
    }
    maxval = warpReduceMax(maxval); //一个warp当中的最大值 ，总共有4个warp，也就是有4个最大值。
```

1. 随后，我们将**一个 warp 内所有线程归约得到的最大值**写入共享内存中的 `maxvals[warpId]`。这样做的目的是为后续计算整个线程块范围内所有元素的最大值做准备。

```C++
if (laneId == 0) 
    maxvals[warpId] = maxval;
__syncthreads();
```

1. 此时，一个线程块中所有 warp 的局部最大值已分别存储在 `maxvals` 数组中。**因此，只需安排一个线程对这些局部最大值进行最终归约，即可得到该线程块内的全局最大值**，并将其写入 `maxvals[0]`。

```C++
 if (tid == 0) {
     float val = maxvals[tid];
     for (int i = 1; i < warpsPerBlock; i++) {
         val = fmaxf(val, maxvals[i]);
     }
     // store the final max in the first position
     maxvals[0] = val;
}
```

#### 启动函数和性能对比

```C++
int blockSize = 128;
int numBlocks = N;
softmax_forward_kernel2<<<numBlocks, blockSize>>>(d_out, d_inp, N, C);
```

在输入维度为 32×4096 的测试下，优化3的性能表现与优化1接近，远优于优化2和原始实现，体现了共享内存与warp洗牌协同设计的有效性。如果需要执行，还是修改course3/softmax_naive.cu中的第294行。

```Bash
Results match: YES
CPU time: 13.9098 ms
GPU time: 0.049152 ms
Speedup: 282.995x
```

#### block大小对计算效率的影响

在 softmax 的实现中，通常由多个 warp 各自进行局部归约，计算各自的局部和与最大值，最后再对所有 warp 的结果进行一次全局归约，得到最终的和与最大值。因此，线程块大小（block_size）会直接影响执行效率。通常情况下，较大的 block_size（如 1024）能更好地利用 GPU 的并行能力，提升整体吞吐。

以下是我们在block_size等于1024时的运行情况：

```Bash
Results match: YES
CPU time: 13.9241 ms
GPU time: 0.032768 ms
Speedup: 424.93x
```

#### 使用nsight-compute进行分析

这里用到了两个rep文件分别是softmax_4.ncu-rep和softmax_4_optimize.ncu-rep，它们各自是block_size等于128和1024时的情况。我们看它们的对比，当block_size提高以后，SM的计算效率有了非常明显的提升。

换句话说，block_size从128到1024时，访存的基本模式没有发生本质上的变化。但是更多的 warp 意味着每个 block 提供了更高的 **内部并行度**，使得当部分 warp 因访存停顿时，所以调度器有更多机会从 **同一个 block 中选择其他就绪的 warp** 来填满计算单元，从而提高 SIMD 利用率，减少空闲周期。

## 结语

本节系统地介绍了Softmax算子的CPU与GPU实现，并重点探讨了在CUDA平台上通过多种优化手段提升其计算性能的方法。

- 从最初的串行式GPU实现出发，我们逐步引入线程块内归约、共享内存、warp洗牌指令等关键技术，实现了对Softmax中最大值与指数和的高效并行计算。
- 优化1通过共享内存和线程块内归约显著提升了性能；
- 优化2引入warp洗牌指令进一步降低数据通信延迟，但受限于线程块大小必须为32；
- 优化3在此基础上结合共享内存，支持更大线程块内的多warp协作，兼顾了并行度与效率。

实验结果表明，在输入维度为32×4096的情况下，优化1和优化3均取得了约4.3倍以上的加速比，显著优于CPU实现。这些优化策略不仅适用于Softmax，也为其他需要归约操作的并行算法提供了通用的设计思路和实现范式。