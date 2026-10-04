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

## Engine 接收+发送数据

![](/blogs/vllm-offline-online-streaming/12.png)

前端用户请求通过http传给AsyncLLM，后台输入线程运行 `process_input_sockets()`，通过 Poller 等待前端消息，解码后放入 `input_queue`。主循环与输入线程通过这个队列衔接。

![](/blogs/vllm-offline-online-streaming/13.png)

`process_input_sockets` 与 `_process_input_queue` 分别处于 Engine 请求处理链路的不同阶段。

1.前者运行在后台输入线程中，负责通过 ZMQ socket 接收前端发送的请求消息，并在完成反序列化后将其放入内部输入队列 `input_queue`；

2.后者则运行在 Engine 主循环中，负责从 `input_queue` 中取出这些请求，并进一步交由调度器和执行器处理。

因此，`process_input_sockets` 主要承担外部通信与入队的职责，而 `_process_input_queue` 主要承担内部取队列与分发处理的职责。

然后Engine从`input_queue` 获取请求并且开始推理

![](/blogs/vllm-offline-online-streaming/14.png)

- **AsyncLLM 提交请求**：把处理后的 prompt 和生成参数发给 Engine。
- **`process_input_sockets` 接收**：解码消息，放进 `self.input_queue`。
- **`_process_input_queue` 取出**：分发请求，把新增请求交给 Scheduler；随后引擎主循环调度并执行推理。
- **推理结果入队**：引擎主循环把生成结果放进 `self.output_queue`。
- **`process_output_sockets` 发送**：从输出队列取结果，通过 ZMQ PUSH 发回 AsyncLLM 所在的 API 服务进程。
- **AsyncLLM 处理结果**：接收结果、解码成文字，按请求 ID 投递到对应的结果队列，供请求协程读取，最终返回给 curl。

## 八股

#### vllm推理的整个流程？

用户通过 Chatbox、网页或 curl，向 vLLM 的 HTTP 接口发送请求。API 层处理消息并准备模型输入，然后调用 AsyncLLM。

AsyncLLM 为请求注册专属输出队列，通过 EngineCoreClient 提交请求。EngineCoreClient 是通信代理，使用 ZMQ 的 ROUTER→DEALER，把请求发送到独立的 EngineCore 进程。

EngineCore 的接收线程解码请求并放进 input_queue，主循环取出后进行调度和模型推理。结果进入 output_queue，再通过 PUSH→PULL 回传前端。

vLLM 前端进程中的 AsyncLLM收到结果后，OutputProcessor 将 token 转成文本，并按 request_id 分发到请求专属队列。AsyncLLM 的 generate 持续取结果，通过 yield 交给 Serving；开启流式模式时，Serving 用 SSE 持续返回给用户，直到生成结束。