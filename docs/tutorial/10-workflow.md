# 10：把三个 Agent 串起来，用一个日志看完整运行

## 这一章的开发流程与经验

串联 agent 时，先用普通 Python 写清阶段顺序。每个 agent 只负责自己的任务，编排程序负责传递文件、保存状态、预算和停止条件。

1. 定义一轮要经过哪些阶段，每个阶段输入哪个产物。
2. 调用前保存阶段名与预定输出位置，出错后仍能找到日志。
3. 共享一份主日志，保留轮次与 agent 身份。
4. 在阶段预算之外增加整次运行的调用预算。
5. 反馈写入成功后再开始下一轮，重新准备上下文。
6. 正常结果、没有可执行方案、执行失败分别记录。

经验：一个调用循环并不会自动变成多 agent 流程。需要程序明确“谁先做、谁拿谁的输出、什么时候停止”。完成一轮不等于候选有效，也不等于全部轮次成功。

## 1. 本章有哪些组件

| 组件 | 输入 → 输出 | 代码 |
| --- | --- | --- |
| iterate 命令 | 父模型、轮数和配置 → 整次报告 | [cli.py](../../src/mini_agentx/cli.py) |
| 编排器 | 上一阶段产物 → 下一阶段调用 | [workflow/runner.py](../../src/mini_agentx/workflow/runner.py) |
| BudgetClient | messages → LLM响应，并计数、保存请求 | runner.py |
| 主日志 | 所有阶段的动作 → 单份可读时间线 | [activity_log.py](../../src/mini_agentx/agents/activity_log.py) |
| 三个 agent | 调查提案、开发训练、评价观测 | 前面的06、07、05章节 |
| 反馈 | 每轮结果 → 下一轮查询记录 | 第09章 |

对应 [AgentX §3](https://arxiv.org/html/2606.26859v2#S3) 的阶段协作目标。这里采用串行多轮、固定父模型作为教学实现；没有工业并行调度、自动晋升或 SGPO。

## 2. 一轮到底怎么接

```text
build_context：父模型 + 任务 + 是否有反馈
    ↓
Brainstorm：调查工具 → 选中方案 proposal.json
    ↓
Developing：读候选 → 修改 → 检查 → 训练 → handoff.json
    ↓
模型A/B：实际加载 A/B → 排名 → 合成事件与统计
    ↓
Evaluation：统计证据 → 判断、引用、程序复核
    ↓
反馈入库 → 本轮完成
    ↓
下一轮重新 build_context，查询刚写入的反馈
```

每轮都基于命令指定的同一个 baseline，并独立从头训练候选。KEEP 不自动成为新父模型，promoted=false；当前不支持自动冠军更新。多轮的联系来自实验反馈，不是累积修改同一份代码。

## 3. 先看 runner 的控制流程

打开 iterate，找到 for index in range(1, rounds + 1)。每次循环先增加 round_state，保存 status.json，然后构造上下文。不是先做完工作，最后才记录状态。

Brainstorm 返回 ready 才进入开发。如果是 needs_evidence，记录后停止，不强行选方案。Developing 成功后再调用 candidate_evaluation，后者负责 A/B、Evaluation 和反馈。

只有反馈成功保存，本轮才标记 completed。否则停止，避免下一轮在不知道上轮结果的情况下继续。阶段错误记录为 failed，停止余下轮次，没有自动无限重试。

REVIEW_REQUIRED 可以作为明确的待复查反馈进入下一轮；它不被转换成模型无效或已有效。下一轮应该查询并注意限制。

## 4. 多个 agent 为什么能写进同一个文档

每个阶段仍有自己的 ActivityLog。编排器另外创建主日志，通过 parent 传入阶段：

```python
activity = ActivityLog(directory, '完整工作流')
# 阶段内部
child = ActivityLog(stage_directory, 'Developing', parent=activity)
```

child.record 写入本阶段文件后，立即调用 parent.record，并带上原 agent 名称。Evaluation 的子日志也通过 A/B阶段日志继续传到主日志。

主日志保留第几轮、谁在执行、时间、动作、公开目的和完整结果。它实时追加，不需要阶段结束后人工拼接。公开目的来自模型 reason 或固定程序说明，不是内部完整思维链。

## 5. 主日志应该怎样阅读

整次运行入口是 runs/workflow-<id>/activity.md。标题类似：

```text
[第1轮 · Brainstorm] 请求工具：read_feedback
公开说明：查看已有尝试，避免重复未经确认的机制。

[第1轮 · Developing] 请求工具：write_candidate_file
公开说明：按方案修改 L2，保持其他参数不变。

[第1轮 · Evaluation] 模型返回评估
公开说明：模型对观测的判断依据。

[第1轮 · 完整工作流] 本轮完成：修改、观测与反馈
```

以上是格式示例，具体内容以实际日志为准。工具参数和完整JSON放在可展开区域，文件内容、检查状态、关键指标、代码差异等直接展示。过长错误先显示摘要，完整错误仍可展开。

末尾“按轮查看结果”表可以快速看哪些轮次完成、最终判决和反馈 ID，再向上阅读详细动作。也可以在编辑器中搜索“第2轮”或“工具失败”。不同 Markdown 阅读器对 details 支持不同，源码中完整内容一直保留。

对应 activity.jsonl 保存同一批事件，包含 agent、round、action、reason、result。原阶段目录与独立日志也保留，主文件无需只靠链接才能看到工具结果。

## 6. 每次请求到底包含什么

编排器没有增加一个负责指挥的 LLM。三个 agent 继续使用各自章节中的 prompt，只是客户端外包了一层 BudgetClient。

每次 complete 前，BudgetClient 将实际 messages 保存到主运行目录：

```text
round-01-Brainstorm-request-01.json
round-01-Developing-request-01.json
round-01-Evaluation-request-01.json
round-02-Brainstorm-request-01.json
```

主日志记录消息条数和路径。请求仍是各 agent 的 system 规则、user 输入、累积的动作与工具观测；认证 key 不在 messages。主日志展示动作过程，完整请求文件用于逐条学习提示词。

## 7. 如何运行

先完成数据和基线，配置 .env，在项目根目录执行：

```bash
mini-agentx iterate --baseline-run baseline-c30ead6da742 --rounds 1

# 两轮，每轮都查询已有反馈
mini-agentx iterate --baseline-run baseline-c30ead6da742 --rounds 2 --max-model-calls 48
```

这个命令会发送项目候选代码和证据到已配置的 DeepSeek，并执行本地隔离检查、训练和A/B。仍要求 macOS sandbox-exec，不降级执行。

rounds 范围1..5；默认总调用预算为每轮24次，包含Brainstorm最多10、Developing最多12、Evaluation最多2。可用 max-model-calls 设置更小总预算，总上限120。API失败也计数。各agent内部次数、工具次数和训练超时继续生效；总预算不等价于精确金额预算。

## 8. 成功与失败怎样保存

工作流目录含 activity.md/jsonl、status.json、report.json 和每次请求文件。status 保存每轮当前阶段、产物路径、判决、反馈和错误；写入使用临时文件替换。状态仅用于查看，当前没有 resume，不会声称能够从任意步骤自动恢复。

| 状态 | 含义 |
| --- | --- |
| running | 正在执行 |
| completed | 请求的全部轮次完成评估并保存反馈 |
| needs_evidence | 当前没有可执行方案，停止开发 |
| failed | 某阶段失败或预算用尽，后续轮次停止 |

第一轮完成、第二轮失败时，整次状态是 failed，但第一轮产物与反馈保留。early Brainstorm/Developing 失败目前不统一写成实验反馈，原日志仍可查；第09章只覆盖进入候选评估阶段的反馈。

## 9. 本地演示与练习

真实两轮运行 workflow-55e7c6e349b1：第一轮完整经过提案、开发、模型A/B、Evaluation和反馈，判决EXTEND。第二轮读取新反馈，但两次提案校验失败，工作流在Brainstorm停止。共23次实际模型调用，主日志包含两个轮次、三个agent、A/B环境及编排程序动作。

这个例子展示正常交接与失败停止，不是“两轮全部成功”。失败原响应和错误都保留，不能为了展示成功删除。

练习：从主日志找到第一轮反馈 ID，检查第二轮 context 与 read_feedback 结果；再找到第二轮校验错误，解释为什么没有进入 Developing。

```bash
python -m unittest discover -s tests -v
```

测试覆盖两轮阶段顺序、每轮重新构造上下文、总预算耗尽停止、无 ready 不进入开发、子日志保留轮次和agent。mock 两轮验证流程，真实调用演示单独记录。

下一步可以实现按已保存阶段恢复、更多失败记忆和候选晋升协议，再考虑 SGPO；不需要先增加新agent。
