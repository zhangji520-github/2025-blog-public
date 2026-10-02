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

## 宽加载的前提和代价

- 地址满足向量类型的自然对齐要求；`float4` 对应 16 字节对齐，子矩阵偏移可能打破它。
- 尾部至少还有 4 个有效元素，不能让宽加载跨出合法范围。
- 输出写回同样要检查地址、行间距与剩余列数，不能只优化输入。
- 更多预加载值会占用寄存器；更少的指令不一定能抵消寄存器压力和较低占用率。

我的判断方式是：先确认访问地址正确，再看生成的指令、实际请求和资源使用，最后看 Kernel 耗时。向量化是一种改善数据搬运组织的方式，不是固定倍数的加速承诺。

## 学习来源

- [向量化访存的原始学习资料](https://tvle9mq8jh.feishu.cn/docx/GpIkdSRagoQur1xmbIYclR2Xngf)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [NVIDIA：An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
- 相关：[矩阵乘法的分块与数据复用](/blog/cuda-matmul-tiling)
