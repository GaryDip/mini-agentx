# 06：Brainstorm 的输入输出契约

## 论文对应

先读 [AgentX §4.1–4.4](https://arxiv.org/html/2606.26859v2#S4)。我们把任务边界、证据、候选成熟度与交接转换为 JSON 契约。代码使用 Python dataclass 和显式校验，这是本地工程选择；不实现原论文动态证据权重或工业人工审查。

本小节介绍契约与校验；DeepSeek 生成流程已在 [下一节](06-brainstorm-agent.md) 接入，不改模型或运行候选实验。

## 输入：TaskBoundary

配置：`configs/brainstorm-task.json`。

| 字段 | 含义 |
| --- | --- |
| schema_version、task_id | 格式版本和任务身份 |
| objective、primary_metric | 优化目标与固定验证指标 |
| allowed_files | 独立候选目录内可改的 model.py、training.toml |
| forbidden_changes | 不可改的数据、评估器、模拟器和密钥等 |
| constraints、unknowns | 已知约束和显式未知信息 |
| candidate_count | 每批 3 个候选，首版最多 5 个 |
| max_training_seconds | 下一阶段训练工具执行的预算，默认 60 秒 |

此处的白名单是候选目录中的相对路径，不是允许直接覆盖仓库基线。Developing 未实现，文本约束与预算目前仅记录，尚无执行器保障；后续要在工具层落实。

## 上下文：证据与父实验

```bash
mini-agentx brainstorm-context --baseline-run <baseline_run_id>
```

本机可使用 `baseline-c30ead6da742`。命令校验权重摘要，从报告白名单取出事实，保存 `runs/brainstorm-input-<id>/context.json`。

context 包含任务、父实验、数据版本、evidence 与 avoid_set。每条证据包括 ID、来源、状态和 facts：

- system-model：模型、维度、特征和指标说明，documented。
- baseline-validation：真实验证指标、最佳轮次和耗时，observed。
- popularity-validation：热门推荐的真实验证指标，observed。

不把整个报告传给模型；没有测试结果、API 配置、原始用户或私有模拟参数。data_analysis/model_research 来源已留在契约，但尚无实际证据接入。avoid_set 初始为空，后续由失败记忆填充。

## 输出：ProposalBatch 与 Proposal

批次携带 schema_version、task_id、parent_run_id 和 proposals。每个提案有：

| 字段 | 用途 |
| --- | --- |
| proposal_id、title | 提案身份与标题 |
| mechanism_key | 机制标识，用于显式去重和 avoid_set |
| maturity | ready / probe_first / backlog |
| priority | 正整数，越小越优先 |
| hypothesis | 需要验证的因果假设，不是已经实现的收益 |
| evidence_refs | evidence_id、field、value，引用已有事实 |
| changes | file、description，修改计划 |
| validation | 指标、方向、同数据同种子比较、观测与证伪条件 |
| risks、probes | 风险和阻塞性调查 |

ready 必须有修改计划，不能还有阻塞性 probes；probe_first 必须说明调查内容。校验后只交接优先级最高的 ready；没有 ready 则 needs_evidence，不强行进入开发。

## 可运行的手写示例

```bash
mini-agentx check-proposals --context examples/brainstorm/context.json --proposals examples/brainstorm/proposals.json
```

示例是手写教学数据，证据标记 synthetic_example，父实验与数据版本也标明 synthetic。三条候选分别展示物品偏置（ready）、负采样诊断（probe_first）、内容特征（backlog）。它们不是 LLM 输出，也没有任何效果已被验证。程序输出示例交接，不会执行或训练。

## 校验能做什么

未知字段、版本错误、父实验不一致、候选数量错误、路径越界、非白名单文件、不存在或错误的证据引用、改变评估指标、重复 ID/机制/优先级、缺调查说明都会被拒绝。批次有无效候选时整批拒绝，下一阶段允许 LLM 有限修复。

mechanism_key 只做精确匹配，不能判断语义重复。程序核对引用值，不证明证据支持假设，也不能阻止白名单文件内出现错误代码；这些需要后续语义审查、工具验证和真实实验。

## 逐步骤论文对齐

| 步骤 | 对应及类型 | 代码 | 输入 → 输出 | 保留和差异 |
| --- | --- | --- | --- | --- |
| 任务边界 | §4.1；简化 | TaskBoundary | 请求 → 明确范围 | 人工配置，尚未自动澄清 |
| 证据上下文 | §4.3；简化 | build_context | 基线 → 有来源的事实 | 三条本地证据，无动态加权检索 |
| 成熟度与批次 | §4.2；简化 | Proposal/validate_batch | 三候选 → 分类 | 契约已实现，生成循环未接 |
| 交接 | §4.4；简化 | validate_batch | 合规 ready → selected_proposal | 程序筛选，不替代生产人工审核 |
| 可证伪计划 | §5.2.1–5.2.2；简化 | validation | 假设 → 观测与证伪条件 | 当前文字计划，后续落实训练日志 |

## 验证与下一步

契约测试覆盖无 ready、越界路径、幻觉引用、指标修改、重复机制、缺 probes、布尔优先级及 avoid_set。真实基线上下文已生成。生成调用、有限修复及持久化交接见 [Brainstorm Agent](06-brainstorm-agent.md)。
