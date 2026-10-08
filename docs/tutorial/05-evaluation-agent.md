# 05：Evaluation Agent——从实验证据到可核查判断

## 论文对应

对照 [AgentX §6 与附录 C.3](https://arxiv.org/html/2606.26859v2#S6)：让 LLM 将观测证据转成结构化判断、解释和后续建议。环境沿用上一章合成 A/B；本章不是线上收益复现，也还不是完整的工具自主调用 agent。

使用 [DeepSeek 官方接口](https://api-docs.deepseek.com/) 和 [JSON Output](https://api-docs.deepseek.com/guides/json_mode/)；默认 deepseek-flash，可通过 DEEPSEEK_MODEL 覆盖。JSON 模式只保障输出格式，不能代替 schema、事实或决策校验。

## 运行

项目根目录的 `.env`：

```dotenv
DEEPSEEK_API_KEY=你的实际密钥
```

环境变量优先于 .env。模板 `.env.example` 不含密钥；请勿提交真实 .env。

```bash
python -m pip install -e .
mini-agentx ab-evaluate --run <experiment_id>
```

也可设置 `--llm-config configs/llm.toml`。当前每次评估最多 2 次模型调用，每次输出上限 2,500 token、网络超时 45 秒；HTTP 认证/余额等不可恢复 4xx 直接停止，429、网络故障或输出校验失败可在预算内重试。失败不会产生有效结论，也不会自动采用规则答案伪装 LLM 成功。

调用向 DeepSeek 发送统计摘要、政策与模拟标识，不发送 API 密钥正文、原始用户记录或 simulator_private.json。密钥仅用于 HTTPS 认证 header，不进入 trace。

## 一次评估如何执行

```text
读取报告并核对事件摘要
 → 去掉规则答案与私有场景信息
 → system prompt + 观测证据
 → DeepSeek JSON 响应
 → schema 与证据引用校验
 → 与程序规则参照比较
 → 保存结论/分歧与轨迹
```

程序目前只读取 report.json 和 events.csv，不给 LLM 文件访问工具。模型看不到私有场景文件，也看不到 rule_decision；它依据 policy 与观察自行得出 KEEP/EXTEND/DISCARD。

结构化输出包含 simulated、verdict、rationale、citations、risks、next_steps、caveats。
每条 citation 必须引用真实点分隔路径及对应值。必须覆盖效果、置信区间、护栏、样本量和流量检查。数字引用不匹配或缺字段触发有限修复。

程序能验证结构化引用，**不能保证自由文本解释中的每个推断都正确**。解释仍需审阅，不应把 LLM 对机制的猜测记为已证实归因。

LLM 判决与规则参照一致时得到对应 final 判决；分歧标记 REVIEW_REQUIRED，不自动发布、回滚或扩展流量。规则参照来自上一章固定政策，不是人类标签，也不能用匹配率证明模型在真实业务中可靠。

## 产物

每次调用独立保存在 `runs/<ab-id>/evaluations/<evaluation-id>/`：

- evidence.json：实际发送的盲评证据。
- trace.json：提示词、调用尝试、响应、usage、延迟或脱敏错误。
- assessment.json：LLM 判断、引用校验、规则对照与最终状态；仅成功评估产生。

原始 A/B 报告和事件不会被修改；重跑评估新建目录。usage 记录消耗的 token，暂不估算费用。没有通过 API 的测试明确使用 mock，不计为真实验证。

## 逐步骤论文对齐

| 步骤 | 对应及类型 | 代码 | 输入 → 输出 | 简化与省略 |
| --- | --- | --- | --- | --- |
| 接口与密钥 | §3；工程支撑 | llm/client.py | 环境配置 → API | DeepSeek 是本地选择 |
| 证据交接 | §6.3；简化 | evaluation_evidence | 统计 → 盲评上下文 | 无线上数据查询 |
| LLM 判断 | §6、附录 C.3；简化 | agents/evaluation.py | 证据 → 结构化解释 | 单轮推理，未接自主分析工具 |
| 引用校验 | §5.2 的客观事实原则；工程支撑 | validate_output | JSON → 校验结果 | 检查引用而非全面证明自由文本 |
| 规则与分歧 | §6.3；简化 | final | 两类判决 → 有效/需复查 | 无复杂业务护栏和人工例外路径 |
| 轨迹保存 | §3、§6.4；简化 | evaluations/ | 尝试 → 文件 | 尚未写入 SQLite 记忆 |

## 按代码一步一步实现

本章依赖 [模拟 A/B 环境](04-ab-environment.md)。环境负责产生事件、统计与规则参照，本章负责让 LLM 分析证据并保存经过校验的结论。环境与 agent 各有一个教程入口，避免把统计计算和模型判断混在一起。

### 第一步：固定证据来源与完整性

入口 `agents/evaluation.py:evaluate_experiment(run, client)` 读取 report.json，要求 simulated=true，再计算 events.csv 的 SHA256 与报告摘要核对。不匹配就停止，尚未调用模型；这是防止统计报告与事件文件被混用的工程检查，不是对整个实验可信度的全面证明。

随后 `abtest/simulator.py:evaluation_evidence` 从报告挑选观测、政策和模拟标识，去掉 rule_decision。simulator_private.json 不读取。这使 LLM 可以根据事实做判断，而不是抄程序的结论。

学习检查：打开 evaluations 下的 evidence.json 与原 report.json，比较哪些字段进入模型，确认没有规则答案与私有场景参数。

### 第二步：构造 messages 与调用客户端

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)},
]
candidate, metadata = client.complete(messages)
```

上面是实际调用的核心形状。SYSTEM_PROMPT 在 evaluation.py，指定角色、规则、模拟限制和输出字段；证据是 user 消息中的数据。`llm/client.py` 发一次 HTTPS 请求并解析 JSON，metadata 返回模型名、usage 和延迟。外层 evaluate_experiment 决定是否再次调用；客户端内部没有额外重试循环。

这里使用固定证据分析流程，**Evaluation 当前没有自主工具调用循环**。LLM 不会自己读取文件或启动 A/B；调用 ab-evaluate 时，由 Python 先读取实验，然后把摘要发送给模型。Brainstorm 的主动调查工具循环见 [06](06-brainstorm.md)。未来可以为 Evaluation 增加读取分组诊断的工具，但现在不能把它描述成已实现。

### 第三步：设计可校验的响应

| 字段 | 下游用途 |
| --- | --- |
| simulated | 必须为 true，维持模拟语境 |
| verdict | KEEP、EXTEND 或 DISCARD |
| rationale | 非空的中文判断依据 |
| citations | 点分隔证据路径与对应原始值 |
| risks、next_steps、caveats | 非空说明列表，记录风险、行动建议和限制 |

与 Brainstorm 引用 facts 的直接字段不同，这里使用嵌套路径，例如 `analysis.primary.ci95`。`resolve_path` 逐层查找字典；`values_match` 检查值、类型与有限数字，数值比较使用很小的浮点容差，列表递归核对。布尔值不会被当作 0/1 数字混用。

`validate_output` 强制引用覆盖效果、区间、护栏、样本量和流量平衡五类证据。它检查必需字段和内容，并非全字段的严格拒绝未知键 schema。JSON 能解析只完成第一层，引用能核对完成第二层，解释的逻辑是否成立仍需审阅。

### 第四步：把错误反馈给模型，有限修复

若 schema 或引用校验失败，程序保存原响应和错误，再追加两条消息：assistant 的错误响应，以及 user 的明确校验错误。下一次 complete 能根据同一原始证据修复输出。

最多两次尝试包括 API 失败和无效输出。不可恢复 HTTP 4xx 直接停止，429 和网络故障可在剩余次数内重试。最终没有合法响应时保存 trace 并抛出失败，不生成 assessment.json，也不把规则答案伪装成 LLM 结论。

### 第五步：校验通过后比较规则参照

程序现在才读取 `report.analysis.rule_decision`，与 LLM verdict 比较：相同则 final 保留该 verdict；不同则 final=REVIEW_REQUIRED。这是模型解释与固定政策之间的分歧检查，不是人工审核通过，也不是自动生产部署。KEEP 等状态只是本地判断产物，当前不会修改真实流量或执行回滚。

### 第六步：把判断保存成下游资产

成功产生 assessment.json，包含 llm_assessment、rule_decision、validation 和 final。完整闭环将来可读取这些结构化字段，把失败与结论存入实验记忆；目前没有 SQLite 或下一轮自动反馈。

本章目前使用 evidence.json、trace.json、assessment.json 查看输入、响应与结果；尚未接入 Brainstorm 的统一 activity.md 日志。

## 运行与学习验收

```bash
mini-agentx ab-evaluate --run <你的ab实验ID>
python -m unittest discover -s tests -v
```

先运行上一章的 ab-run 得到 ID。一次评估后，依次读 evidence、trace、assessment：追踪一条 citation，确认值来自实际证据；核对 verdict 与 final 是否相同，再解释为什么一致或转复查。

离线测试覆盖幻觉数值拒绝、有限修复、上下文排除规则答案与模拟真值、判决分歧转复查、事件被篡改时不调用模型。测试使用 mock；下面的真实 API 记录单独列出，两者不混用。

下一步是 Developing 的代码与训练工具，再由编排器连接 Brainstorm → Developing → Evaluation。未来还需要固定候选晋升协议、诊断工具和实验记忆；当前不应声称三个 agent 已经组成完整闭环。

## 真实 API 验证记录

使用本地 .env 调用 DeepSeek，三种观测各成功调用一次，结构化引用与判决校验通过。沙箱网络失败的首次尝试另存为失败轨迹，不计成功。

| 观测 | LLM 判断 | 引用校验 | 规则一致 |
| --- | --- | --- | --- |
| 模拟改善 | KEEP | 通过 | 是 |
| 模拟恶化 | DISCARD | 通过 | 是 |
| 样本不足 | EXTEND | 通过 | 是 |

这只是三个教学样例，不是完整准确率评测。真实响应的自由文本里也出现泛化的方差提醒；我们已经按用户聚类 bootstrap，不能据此认定程序将重复会话当作独立样本。此类表述需人工核查或后续加强事实校验。
