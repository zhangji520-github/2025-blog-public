# Vllm 的离线在线推理部署

## 离线推理

打开 `code/course0/offline.py`，代码按以下要点组织：

**要点 1：**本课程以 vLLM 0.30.0 的 V1 引擎为主。离线 `LLM` 可通过设置 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 将前端与 EngineCore 放在同一进程，便于调试。

**要点 2：**定义四组提示文本，作为用户输入供模型续写：

```Python
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
```

**要点 3：**采用采样参数 `sampling_params` 控制生成行为，其中包含两个关键参数。本节课仅作简要理解：

- **温度（temperature）**：用于调节模型输出的概率分布多样性。温度越高，输出分布越平缓，随机性越强；温度越低，分布越尖锐，倾向于选择高概率词元。当温度趋近于 0 时，模型输出趋于确定性，几乎总是选择概率最高的词元。
- **Top-p（nucleus sampling，p=0.95）**：按概率从高到低累加候选词元，保留累计概率达到或超过 p 的最小集合，再从中采样。它限制了低概率候选参与采样，但不保证生成内容的语义正确性。

![](/blogs/vllm-offline-online-streaming/01.png)

**要点 4：**`llm = LLM(model="/root/model")` 使用前面下载的 Qwen3-0.6B 本地模型；也可以将 `model` 设为模型仓库标识。初始化时，vLLM 加载权重并规划 KV Cache 等运行时显存。KV Cache 保存历史 token 的键和值，供后续解码复用。

**要点 5：**`outputs = llm.generate(prompts, sampling_params)` 提交请求并等待生成完成。默认多进程模式下，`LLM` 初始化时会建立前端 `LLMEngine` 与独立 `EngineCore` 进程的通信：

- **LLMEngine** 处理输入并通过 `EngineCoreClient` 向 EngineCore 提交请求，随后接收内部结果。
- **EngineCore** 维护调度状态，通过 Executor、Worker 和 ModelRunner 执行模型计算，并将结果返回前端。

LLMEngine 在启动时，会根据当前运行模式创建一个通往 EngineCore 的客户端句柄。此后，所有推理请求都会通过 `self.engine_core` 发送给底层执行引擎。也就是说：

- LLMEngine 主要负责对外接口层的工作，例如 tokenizer、输入预处理、输出后处理、统计日志、trace、兼容旧接口等。
- EngineCoreClient 负责将 LLMEngine 的请求转交给真正的底层引擎 EngineCore，并将执行结果返回给上层。

这里的 `self.engine_core` 是 `LLMEngine` 持有的客户端对象。在默认多进程模式下，EngineCore 本体运行于独立进程；设置 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 时，客户端改用同进程实现。

## 在线推理

1. 先在launch.json写好配置

![](/blogs/vllm-offline-online-streaming/02.png)

1. 启动服务
2. 可以在终端curl我的端点打上请求

![](/blogs/vllm-offline-online-streaming/03.png)

**`model`**：选择服务提供的模型，名字要和启动配置中的 `--served-model-name` 一致。

**`messages`**：发送聊天记录。这里 `role: "user"` 表示用户说的话，`content` 是“你好”。

**`max_tokens`**：最多生成 128 个 token，不等于 128 个字；模型也可能提前结束。

## 流式输出

“用户发问题 → 模型生成 → 返回回答”

可以把 vLLM 的在线服务理解为基于 `AsyncLLM` 的前端与 EngineCore 协作架构：

- `AsyncLLM` 通过 `EngineCoreClient` 与后台运行的 EngineCore 进程通信；
- 前者负责接收请求、管理异步流并向上层持续返回结果；
- 后者负责调度并调用执行层完成模型推理，将生成的 token 等内部结果返回前端；前端的 `OutputProcessor` 再将其转换为 `RequestOutput`，供上层持续读取。

> 注：EngineCore 内部还需将计算结果逐层传递以完成完整的前向传播，上图聚焦进程间的输出回传链路

![](/blogs/vllm-offline-online-streaming/04.png)

```Plain Text
用户（curl / 网页 / 聊天软件）
          │ HTTP 请求：问题、模型名、生成参数
          ▼
      API Server ： 接收 HTTP 请求、检查参数，再把结果包装成 HTTP 响应
          │ 调用
          ▼
       AsyncLLM ： 管理异步推理请求，处理输入，并把引擎输出整理成上层可用的结果
          │ 通过内部客户端提交请求
          ▼
    EngineCoreClient ： AsyncLLM 使用的内部联络员，负责向 EngineCore 发请求、接收结果
          │ ZMQ 跨进程传递
          ▼
       EngineCore ： 推理执行中心，决定这一轮处理哪些请求，并协调模型执行
          │ 调度请求、管理 KV Cache
          ▼
 Executor / Worker / ModelRunner
          │
          ▼
         GPU
```

### 断点调试

发送http请求给api server

```C++
curl -N --noproxy '*' http://127.0.0.1:13311/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 32,
    "stream": true
  }'
```

用 5 个断点观察“收到问题 → 发给引擎 → 收到 token → 转成文本 → 返回用户”

1. 接到用户请求

![](/blogs/vllm-offline-online-streaming/05.png)

![](/blogs/vllm-offline-online-streaming/06.png)

这证明 HTTP 请求已经进入 API 服务。按 **F5** 继续。

1. 看发给 EngineCore 引擎的请求

![](/blogs/vllm-offline-online-streaming/07.png)

![](/blogs/vllm-offline-online-streaming/08.png)

1. 看enginecore返回的token

![](/blogs/vllm-offline-online-streaming/09.png)

- `request_id` 对应该请求，内部标识可能经过处理。
- `new_token_ids` 是新生成的 token 整数列表。
- `finish_reason` 通常在生成过程中为 `None`，结束时变为停止或长度限制原因。

此处主要是 **token 数据，还不是最终显示的文本**。

这个断点会反复命中。看清一次后，取消或禁用它，按 **F5**。

1. 看 token 转成文本后的结果

![](/blogs/vllm-offline-online-streaming/10.png)

![](/blogs/vllm-offline-online-streaming/11.png)

模型可能先生成思考内容，因此不保证第一段就是“你好”。

这里的 `yield out` 将结果交给聊天接口层。看完后禁用断点，按 **F5**。

1. 看真正发给用户的流式数据

```Markdown
① request.messages：用户的问题
           ↓
② prompt_token_ids：模型的输入
           ↓  EngineCore 执行
③ new_token_ids：新生成的 token
           ↓  输出处理
④ out.outputs[0].text：文字片段
           ↓  接口包装
⑤ data：发给用户的 JSON
```