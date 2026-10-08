# 06：运行 Brainstorm Agent

## 论文对齐

对照 [AgentX §4.1–4.4、附录 C.1](https://arxiv.org/html/2606.26859v2#S4)：任务边界、证据、批次候选、成熟度和交接形成一次真实 LLM 生成流程。本版只有本地基线证据，不实现工业 SQL、外部论文检索、动态证据权重或完整多批探索；失败记忆尚未自动填入 avoid_set。

## 运行

```bash
mini-agentx brainstorm --baseline-run <baseline_run_id>
```

本机可使用 `baseline-c30ead6da742`。命令从基线生成新的 context，然后调用 DeepSeek。也可传 `--context <context.json>` 使用已准备上下文；两种输入互斥。

默认使用 `configs/brainstorm-task.json` 与 `configs/brainstorm-llm.toml`，可用 `--task`、`--llm-config` 覆盖。密钥仍由 .env 读取；现在启用只读调查工具，每批最多 8 次模型调用、6 次工具调用，每次输出最多 4,500 token；无效输出最多两次。这里不修改模型、不训练、不启动 A/B。

工具协议、范围与最新真实轨迹见 [主动调查工具](06-brainstorm-tools.md)。

## 调用链

```text
基线报告 → 白名单证据 → task/context
 → system prompt + evidence → DeepSeek JSON
 → validate_batch
 → 有效：保存候选及交接
 → 无效：反馈明确校验错误，预算内修复
 → 仍失败：保存失败轨迹，停止
```

prompt 在 `agents/brainstorm.py`，包含角色、事实与假设的区分、文件职责、成熟度和 JSON 示例。模型可返回全是 probe_first/backlog；此时合法保存 needs_evidence，不能强行选出 ready。

代码上下文明确：model.py 负责 embedding 与打分；training.toml 只改变现有训练参数；BPR 损失和负采样算法属于不可改的训练器。当前范围不足以实施的方向应放 backlog。

## 产物

- activity.md：统一的时间顺序日志，展示动作、公开行动理由、工具结果、校验与交接；activity.jsonl 是对应的结构化事件。
`runs/brainstorm-<id>/`：

- context.json：实际输入，不含测试指标或私有模拟参数。
- trace.json：system prompt、每次响应、校验错误、模型、usage 和延迟；修复请求可由错误及响应重建。
- tools.json：主动工具请求、结果与错误，context.json 包含查询后追加的证据。
- proposals.json：通过契约的整个候选批次。
- handoff.json：筛选结果、父实验和数据版本。
- proposal.json：仅有 ready 时保存，携带任务、父实验、数据版本和选中提案。
- status.json：ready、needs_evidence 或 failed；失败时没有有效提案产物。

这些文件留在本地，不提交运行轨迹、密钥或模型权重。

## 逐步骤论文对齐

| 实现 | 对应及类型 | 代码 | 输入 → 输出 | 差异 |
| --- | --- | --- | --- | --- |
| 构造输入 | §4.1、§4.3；简化 | build_context | 基线 → 明确边界和事实 | 无自动用户澄清和完整 KB |
| 小批生成 | §4.2；简化 | generate_proposals | context → 三候选 | 单批，未实现残余探索循环 |
| 成熟度与顺序 | §4.2；简化 | prompt/validator | 候选 → ready/probe/backlog | 分类质量仍需审查 |
| 校验与修复 | §4.4；简化 | validate_batch | JSON → 合规/错误 | 规则检查，不替代生产审核 |
| 交接与轨迹 | §3、§4.4；简化 | proposal.json/trace.json | 提案 → 下游产物 | Developing 尚未接入 |

## 真实调用中发现的限制

首次生成通过 JSON 校验，但把负采样修改放进 model.py；实际采样位于固定训练器，不能在该文件实施。开发审查将该批次标记需复查，并补充文件职责及可改参数后重新生成。

这说明结构化校验能验证路径和引用，不能证明修改计划可执行。当前 ready 表示模型判断并通过契约，不等于已通过代码语义审查。Developing 下一步仍须读取真实代码、检查接口和实施位置，再执行训练。

## 验证

18 项项目测试通过。新增测试覆盖无效提案有限修复、无 ready 不产生执行交接、预算耗尽保留失败轨迹。mock 测试不冒充真实 API 调用。提案只是待验证假设，不声称已有收益。

## 最新真实 API 验证

使用真实基线 `baseline-c30ead6da742`、DeepSeek API 成功生成三个 ready 候选：维度 32→64、降低学习率、增强 L2。选中维度调整，修改位置为候选 training.toml，成功批次仅调用一次。输出目录为 `runs/brainstorm-1ff79cfe899c/`。本地其他重跑将生成新 ID 和可能不同的候选。

此前一批输出把数值引用写成字符串，两次校验均失败；明确原始 JSON 类型并将校验错误细化到证据字段后，重新验证通过。失败轨迹保留，不能计作成功。

已选提案路径：`runs/brainstorm-1ff79cfe899c/proposal.json`。其中包含任务、父实验、数据版本、假设、证据、修改计划和证伪方法。没有训练该候选，未产生任何改进收益。
