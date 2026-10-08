# 05：DeepSeek Evaluation Agent

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

## 离线检查

```bash
python -m unittest discover -s tests -v
```

12 项测试通过，新增覆盖数值引用幻觉拒绝、无效输出修复、上下文排除答案与真值、判决分歧转复查、事件被篡改时不调用模型。

## 下一步

Brainstorm：根据任务边界、基线与实验记忆生成提案。随后接 Developing 的代码/训练工具，再串成完整闭环；Evaluation 可进一步升级为按需查询诊断工具的工作流。

## 真实 API 验证记录

使用本地 .env 调用 DeepSeek，三种观测各成功调用一次，结构化引用与判决校验通过。沙箱网络失败的首次尝试另存为失败轨迹，不计成功。

| 观测 | LLM 判断 | 引用校验 | 规则一致 |
| --- | --- | --- | --- |
| 模拟改善 | KEEP | 通过 | 是 |
| 模拟恶化 | DISCARD | 通过 | 是 |
| 样本不足 | EXTEND | 通过 | 是 |

这只是三个教学样例，不是完整准确率评测。真实响应的自由文本里也出现泛化的方差提醒；我们已经按用户聚类 bootstrap，不能据此认定程序将重复会话当作独立样本。此类表述需人工核查或后续加强事实校验。
