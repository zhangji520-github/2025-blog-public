# vLLM 推理链路：从离线批量调用到在线流式返回

vLLM 的离线调用与在线服务共享一个关键思路：前端负责把用户输入整理成请求，EngineCore 负责调度与模型执行，输出处理再把生成的 token 变成调用方可消费的结果。两种入口的差异，主要在接口形态和结果交付方式。理解这条链路后，部署和调试时就能区分 HTTP、引擎与解码三个环节。

## 离线调用：同步取得一批请求的结果

离线批量推理通常从 `LLM` 开始。下面的示例使用一个已下载的模型目录；`model` 也可换成支持的模型仓库标识。模型文件、依赖和 GPU 显存需要事先准备好。

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The capital of France is",
]
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=64)

llm = LLM(model="/path/to/model")
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.prompt, output.outputs[0].text)
```

`temperature` 调整采样分布：值较低时更偏向高概率 token，值较高时分布更平缓。`top_p=0.95` 则按概率从高到低选取累计概率达到阈值的候选集合，再在其中采样。两者控制的是生成选择过程，不能保证事实正确。`max_tokens` 限制生成 token 数量，也不等同于字数。

创建 `LLM` 会加载权重并准备执行资源，其中包括为 KV Cache 规划空间。解码新 token 时，KV Cache 复用历史位置的键和值，避免每一步都重新计算整段上下文。`generate` 接收一批提示，返回每个请求的生成结果；它是调用方等待整批结果的同步接口。

## 同步接口背后的执行分工

在 vLLM V1 的常见多进程配置中，离线 `LLM` 的前端持有 `LLMEngine`，通过 `EngineCoreClient` 与后台的 EngineCore 通信。客户端负责提交请求并取得引擎输出，EngineCore 则维护调度状态，协调 Executor、Worker 和 ModelRunner 执行模型计算。前端的输出处理将内部 token 结果整理为 `RequestOutput`。这也解释了为什么 `LLMEngine` 中的 `self.engine_core` 是通信客户端，并不意味着 EngineCore 本体必定在同一个进程。

```text
Python 代码 → LLM / LLMEngine → EngineCoreClient
                                  ⇅
                              EngineCore → Executor / Worker / ModelRunner → GPU
                                  ↓
                         token 结果 → 输出处理 → RequestOutput
```

`VLLM_ENABLE_V1_MULTIPROCESSING=0` 可让同步引擎选用进程内客户端，便于观察调用链；默认的多进程路径使用后台进程和进程间通信。这个开关描述的是同步 `LLMEngine` 的调试方式，不应直接推广到在线 `AsyncLLM`：当前 V1 实现中的异步客户端要求多进程模式。具体类名与调用位置会随 vLLM 版本变化，排查问题时应以实际安装版本的源码为准。

## 在线服务：HTTP 入口加异步结果流

在线服务在引擎前面增加 API Server。它接收 HTTP 请求，校验参数，并将引擎结果封装成兼容 OpenAI 接口的响应。一个基本启动和请求示例如下：

```bash
vllm serve /path/to/model --served-model-name demo-model --host 127.0.0.1 --port 8000

curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"demo-model","messages":[{"role":"user","content":"你好"}],"max_tokens":32,"stream":true}'
```

请求中的 `model` 要与服务暴露的名称匹配。`messages` 是对话输入；`stream: true` 请求逐步返回结果。`curl -N` 关闭客户端输出缓冲，方便在终端观察响应片段。模型还必须支持所使用的聊天模板，否则应改用适合该模型的输入接口或配置模板。

V1 在线链路可以概括为：API Server 将请求交给 `AsyncLLM`；它通过异步的 `EngineCoreClient` 把请求送往 EngineCore，并持续接收引擎输出。EngineCore 调度请求、管理 KV Cache 并调用执行层。返回的 `new_token_ids` 仍是整数 token ID，前端 `OutputProcessor` 才将其解码并整理为上层的 `RequestOutput`；API Server 再封装为发送给客户端的数据。流式响应的“流”发生在结果持续交付这一层，不代表模型一次前向计算就直接生成完整文本。

```text
HTTP 请求 → API Server → AsyncLLM → EngineCoreClient ⇄ EngineCore → GPU
                ↑                         ↓
          响应数据 ← 文本片段 ← OutputProcessor ← 新生成的 token ID
```

从调试角度，可以沿着五个观察点定位问题：HTTP 层收到的 `messages`，输入处理后的 token ID，EngineCore 返回的 `new_token_ids`，输出处理后的文本，以及最终写给 HTTP 客户端的数据。如果引擎已产生 token 但客户端没有文字，重点检查输出处理与接口封装；如果引擎尚无输出，则继续看排队、调度和执行状态。`finish_reason` 在生成中通常尚未确定，结束时才说明停止原因或长度限制。对于带思考内容的模型，首个流式片段也未必是用户最终看到的正文。

离线接口适合脚本和批量任务：调用方一次提交提示并等待结果。在线接口适合并发请求和交互场景：HTTP 层与异步引擎协作，按请求持续交付结果。两者都依赖同一类核心工作：把请求调度到模型执行，并把内部 token 结果还原为可读输出。

## 资料与版本说明

- 学习资料：[《Vllm 的离线在线推理部署》](https://hqhw9f213hx.feishu.cn/wiki/CQcKwYweOi3hbWkB4W3cJJEmn7d)。本文按知识笔记重组，未复用资料中的课堂截图和未给出完整配置的断点步骤。
- vLLM 官方文档：[EngineCoreClient](https://docs.vllm.ai/en/stable/api/vllm/v1/engine/core_client/)、[AsyncLLM](https://docs.vllm.ai/en/stable/api/vllm/v1/engine/async_llm/)、[OutputProcessor](https://docs.vllm.ai/en/stable/api/vllm/v1/engine/output_processor/)、[OpenAI 兼容服务](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)。

原资料写有“vLLM 0.30.0”，但未附可验证的环境信息。本文不将该版本号作为运行前提；示例与内部实现细节仍应对照实际安装版本核查。
