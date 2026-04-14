# 设计模式

## PipelineUnit 设计原理

DiffSynth-Studio 使用 **PipelineUnit** 模式组织扩散模型的数据预处理流程。

### 设计模式分析

PipelineUnit 架构融合了多种经典设计模式：

| 模式 | 体现 | 作用 |
|------|------|------|
| **责任链** (Chain of Responsibility) | Units 依次执行，各自处理特定任务 | 解耦复杂的数据预处理流程 |
| **数据流** (Dataflow) | 通过字典传递和累积状态 | 灵活组装输入输出 |
| **声明式配置** | Unit 声明 `input_params`/`output_params` | 自描述、可检查 |
| **延迟加载** (Lazy Loading) | `onload_model_names` 控制模型加载时机 | 显存优化 |

### 关键设计决策

**1. 为什么 Unit 只能是前置？**

```
Denoise 循环执行 50+ 次，若 Unit 介入会导致：
- 重复加载 VAE/TextEncoder 等大模型 → 性能灾难
- 计算图复杂化 → 训练时梯度回传困难
- 副作用难以控制 → 调试困难
```

**2. 与标准 Pipeline 模式的区别**

| 对比 | 传统 ML Pipeline | DiffSynth PipelineUnit |
|------|-----------------|----------------------|
| 阶段 | 训练/推理多阶段 | 仅数据预处理阶段 |
| 迭代 | 通常是单向流动 | Denoise 循环在 Unit 之后 |
| 模型 | Pipeline 中直接调用 | model_fn 封装，Unit 只准备条件 |

### 为什么 Unit 不能介入 Denoise 循环

1. **设计解耦**: Unit 通常涉及模型加载（VAE、TextEncoder），而 model_fn 内部是纯张量计算
2. **性能考虑**: Denoise 循环执行 50 步，Unit 若介入会导致重复加载模型
3. **梯度隔离**: 训练时 Loss 计算在 Denoise 阶段，需要清晰的计算图

### WanVideo 调用逻辑

```python
# 1. 初始化 Units（根据配置动态组装）
self.units = [
    WanVideoUnit_ShapeChecker(),
    WanVideoUnit_NoiseInitializer(),
    WanVideoUnit_InputVideoEmbedder(),
    WanVideoUnit_ImageEmbedderVAE(),
    WanVideoUnit_ImageEmbedderCLIP(),
]
if self.enable_text:
    self.units.append(WanVideoUnit_PromptEmbedder())
if self.action_injection_mode != "none":
    self.units.append(WanVideoUnit_ActionEmbedder())

# 2. 前向调用流程
@torch.no_grad()
def __call__(self, prompt, input_video, action=None, ...):
    # 2.1 初始化输入字典
    inputs_shared = {"input_video": input_video, "action": action, ...}
    inputs_posi = {"prompt": prompt, ...}
    inputs_nega = {"negative_prompt": negative_prompt, ...}

    # 2.2 Units 链式处理（关键：只执行一次）
    for unit in self.units:
        inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
            unit, self, inputs_shared, inputs_posi, inputs_nega
        )

    # 2.3 Denoise 循环（执行 N 次）
    for timestep in self.scheduler.timesteps:
        noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep)
        noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep)
        noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        inputs_shared["latents"] = self.scheduler.step(noise_pred, timestep, latents)

    # 2.4 Decode
    video = self.vae.decode(inputs_shared["latents"])
    return video
```
