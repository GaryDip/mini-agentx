# 模拟 A/B：为 Evaluation Agent 建立环境

本章实现的 ab-run 全程由 Python 执行，没有 LLM prompt。要学习模型每一次请求、A/B 证据如何进入 prompt 与错误修复，请先读 [Evaluation 教程的提示词与每次请求](05-evaluation-agent.md#先学-ab-的提示词与每次请求)。


## 论文对应与边界

对照 [AgentX §6.2–6.4](https://arxiv.org/html/2606.26859v2#S6)：保留流量分配、反馈统计、护栏与结构化结论的流程。首版采用固定窗口、合成点击和工程阈值，不实现生产发布、真实用户反馈、CUPED、业务复合收益或人工例外审查。

**本章是环境实现，不调用 LLM。** LLM 接入已在 [下一章](05-evaluation-agent.md) 实现：读取观测证据、解释结果、提出 KEEP/EXTEND/DISCARD，再由程序校验。本章的 rule_decision 是规则参照。

## 数据流

```text
实验配置 → 用户 ID 稳定分桶 → 小流量护栏检查
                              ↓
                        通过后观察完整窗口
                              ↓
曝光/点击/延迟/错误 → 用户级统计 → 证据报告
                              ↓
                    后续 LLM 评估及规则校验
```

一个用户始终在同一组。SHA-256 对 seed 与 user ID 的稳定哈希作 50/50 分桶，不使用 Python 的进程随机 hash。canary 使用另一命名空间的哈希选取约 10% 用户；与 A/B 分桶相互独立。

本地模拟器会预生成完整事件，若 canary 护栏失败，只记录 canary 事件并标记 canary_stopped；未观测的完整流量不进入报告。它是模拟分阶段观察，不是部署平台。

## 先用可控场景学习决策

在项目根目录、mini-agentx Conda 环境执行：

```bash
mini-agentx ab-run --scenario improvement
mini-agentx ab-run --scenario regression
mini-agentx ab-run --scenario guardrail
mini-agentx ab-run --scenario inconclusive
mini-agentx ab-run --scenario null
```

场景里的收益是注入的模拟器参数，用于测试决策，不代表模型获得提升。默认 2,000 个独立用户，每用户 20 次会话；inconclusive 使用 100 个用户、每用户 2 次。

点击概率在用户之间不同；会话生成二项点击与错误，并生成合成延迟。错误请求记为未点击。事件随机流按用户固定，重跑相同 seed/配置得到相同观测与统计，实验 ID 每次新建。

## 接入 A、B 两个真实推荐模型

```bash
mini-agentx ab-run --scenario null --model-a <A的run_id> --model-b <B的run_id>
```

模型模式仅允许 null 场景，避免同时注入人为收益。先核对模型数据版本和权重摘要；两个模型各自生成未见电影 Top-10，每个用户依次轮转曝光列表。实际用户数为训练词表的 942，覆盖所有训练用户，不依赖留出标签。

反馈 oracle 以固定随机电影向量及用户训练历史生成合成偏好，独立于模型的分数和 embedding；相似度映射到有界概率。验证/测试标签不加载（仅核对文件摘要）。A/B 分数只决定排名，不能靠放大分数制造 CTR 提升。

oracle 是明确的教学假设，不是对 MovieLens 真实点击的估计；它与真实用户偏好可能完全不匹配，无法据此证明推荐模型的真实商业价值。

延迟和错误仍是配置生成，尚未实际测量模型推理延迟，也没有模型自行产生的业务错误。

## 统计与规则

配置在 `configs/abtest.toml`，在观测前固定：

- 主指标：每用户 CTR 的组内平均；当前每用户会话数相同，它与总点击/总曝光相同。
- 不确定性：组内重采样用户 1,000 次，得到 B−A 的 percentile bootstrap 95% 区间。重复会话不能当作独立用户扩大有效样本量。
- SRM：对用户数进行预期 50/50 的二项正态近似检查，p < 0.001 时先调查分流，不能简单加流量解决。
- 最小用户数：每组至少 200。
- 最小绝对 CTR 增益：0.005（0.5 个百分点）；CI 下界 >0 且效果超过阈值才 KEEP。
- 负向区间上界 <0 时 DISCARD；证据或样本不足时 EXTEND。
- 简化硬护栏：B/A p95 延迟比 >1.25 或 B 错误率 >2% 时 DISCARD，canary 护栏失败停止升级。

延迟/错误阈值是描述性工程检查，不带置信区间，不能声称护栏统计显著。bootstrap 和 SRM 是固定窗口的教学近似；没有多重检验或序贯检验。EXTEND 是建议，不会自动反复加样本直到显著；继续观察需预先规划新的窗口或后续采用序贯方法。

## 保存的产物与 LLM 边界

`runs/ab-<id>/` 保存：

- events.csv：本次观测的用户、分组、会话、曝光电影、点击、延迟和错误。
- report.json：模拟标识、窗口、策略、分组统计、置信区间、护栏、规则参照与 caveats。
- simulator_private.json：seed、注入场景与参数，只供模拟器复现，**禁止加入后续 agent 上下文**。

读取报告：

```bash
mini-agentx ab-report --run <experiment_id>
mini-agentx ab-report --run <experiment_id> --evidence-only
```

`--evidence-only` 移除完整报告和 canary 分析中的 rule_decision，避免 LLM 仅复述规则答案。注入场景参数未包含在公共报告中。文件分离是上下文约定，不是权限沙箱；后续工具必须限制 agent 对私有配置及文件的访问。

## 实际验证结果

| 场景 | 规则参照 | 实际行为 |
| --- | --- | --- |
| 改善 | KEEP | CTR 绝对增益 0.06570，95% 区间 [0.05548, 0.07484] |
| 恶化 | DISCARD | CTR 增益区间为负 |
| 护栏恶化 | DISCARD | canary 停止，未记完整流量 |
| 样本不足 | EXTEND | 每组用户未满足最小样本 |
| 无效果 | EXTEND | CI 跨零 |
| 同一基线 A/A | EXTEND | CTR 差 0.00046，区间 [-0.01145, 0.01223] |

同一 seed 重跑的事件摘要和统计一致；重复同用户会话不会缩窄用户级 bootstrap 区间。将模型所有 embedding 放大两倍得到相同排名，观测与 A/A 完全一致。九项项目测试通过。

## 逐步骤论文对照

| 步骤 | 对应及类型 | 代码 | 输入 → 输出 | 简化 |
| --- | --- | --- | --- | --- |
| 稳定分桶 | §6.2；简化 | abtest/simulator.py | 用户、seed → A/B | 无工业业务域和流量桶管理 |
| canary 护栏 | §6.2；简化 | simulate | 小流量 → 停止/继续 | 无部署，只模拟观察 |
| 反馈生成 | §6.1 的环境替代 | simulator.py、model_adapter.py | 曝光 → 合成结果 | 非真实线上 reward |
| 统计提取 | §6.3；简化 | statistics.py | 事件 → CI/护栏 | bootstrap 替代生产统计工具 |
| 规则参照 | §6.3；简化 | analyze | 统计 → 三类结论 | 无复合护栏与例外审查 |
| 证据保存 | §6.4；简化 | report.json | 结论 → 文件 | 尚未接 SQLite 记忆 |
| LLM 分析 | §6、附录 C.3；简化 | 下一章 agents/evaluation.py | 证据 → agent 判断 | DeepSeek 单轮评估 |
