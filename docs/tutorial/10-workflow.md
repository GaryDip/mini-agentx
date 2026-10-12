# 10：把三个 Agent 串起来，用一个日志看完整运行

## 这一章的开发流程与经验

串联 agent 时，先用普通 Python 写清阶段顺序。每个 agent 只负责自己的任务，编排程序负责传递文件、保存状态、预算和停止条件。

1. 定义一轮要经过哪些阶段，每个阶段输入哪个产物。
2. 调用前保存阶段名与预定输出位置，出错后仍能找到日志。
3. 共享一份主日志，保留轮次与 agent 身份。
4. 在阶段预算之外增加整次运行的调用预算。
5. 反馈写入成功后再开始下一轮，重新准备上下文。
6. 可信 KEEP 后保存新基线，下一轮使用它的代码和配置。
7. 正常结果、没有可执行方案、执行失败分别记录。

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

对应 [AgentX §3](https://arxiv.org/html/2606.26859v2#S3) 的阶段协作目标。这里采用串行多轮和动态基线作为教学实现；没有工业并行调度或 SGPO。

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

命令中的 baseline 是起始基线。每轮候选与当时的基线比较；可信 KEEP 后，候选的源码、配置和权重被保存为新基线，下一轮在它的源码和配置上继续修改。训练仍从头开始，不加载父权重续训。未采纳时基线保持不变。

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

主日志按轮次和 agent 展示工具调用、公开目的、关键返回、修正与最终决定。原始事件实时追加到 JSONL，Markdown 阅读版根据事件更新。公开目的来自模型 reason 或固定程序说明，不是内部完整思维链。

## 5. 主日志应该怎样阅读

整次运行入口是 runs/workflow-YYYYMMDD-HHMMSS-<id>/activity.md。时间为新加坡时间（UTC+08），例如 workflow-20261012-153045-a1b2c3d4e5f6；按目录名称排序即可比较新运行的先后。末尾随机 ID 避免同秒启动时重名。旧运行目录不重命名，继续可读。标题类似：

```text
[第1轮 · Brainstorm] 请求工具：read_feedback
公开说明：查看已有尝试，避免重复未经确认的机制。

[第1轮 · Developing] 请求工具：write_candidate_file
公开说明：按方案修改 L2，保持其他参数不变。

[第1轮 · Evaluation] 模型返回评估
公开说明：模型对观测的判断依据。

[第1轮 · 完整工作流] 本轮完成：修改、观测与反馈
```

以上是格式示例，具体内容以实际日志为准。阅读版只展示工具参数与关键返回，不再展开整份 JSON、源码或训练历史。保留检查状态、关键指标、配置、代码差异与错误摘要；完整内容在 activity.jsonl。LLM 请求和用量不出现在阅读版。

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

原始 activity.jsonl 记录消息条数和路径，阅读版不展示这些传输信息。请求仍是各 agent 的 system 规则、user 输入、累积的动作与工具观测；认证 key 不在 messages。主日志展示动作过程，完整请求文件用于逐条学习提示词。

## 7. 如何运行

先完成数据和基线，配置 .env，在项目根目录执行：

```bash
mini-agentx iterate --baseline-run baseline-c30ead6da742 --rounds 1

# 两轮，每轮都查询已有反馈
mini-agentx iterate --baseline-run baseline-c30ead6da742 --rounds 2 --max-model-calls 48
```

这个命令会发送项目候选代码和证据到已配置的 DeepSeek，并执行本地隔离检查、训练和A/B。仍要求 macOS sandbox-exec，不降级执行。

rounds 范围1..10；默认总调用预算为每轮52次，包含Brainstorm最多10、三个候选各Developing最多12和Evaluation最多2。可用 max-model-calls 设置总预算，总上限600。API失败也计数。各agent内部次数、工具次数和训练超时继续生效；总预算不等价于精确金额预算。

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

旧版固定基线的真实两轮运行 workflow-55e7c6e349b1（升级功能新增前）：第一轮完整经过提案、开发、模型A/B、Evaluation和反馈，判决EXTEND。第二轮读取新反馈，但两次提案校验失败，工作流在Brainstorm停止。共23次实际模型调用，主日志包含两个轮次、三个agent、A/B环境及编排程序动作。

这个例子展示正常交接与失败停止，不是“两轮全部成功”。失败原响应和错误都保留，不能为了展示成功删除。

练习：从主日志找到第一轮反馈 ID，检查第二轮 context 与 read_feedback 结果；再找到第二轮校验错误，解释为什么没有进入 Developing。

```bash
python -m unittest discover -s tests -v
```

测试覆盖两轮阶段顺序、每轮重新构造上下文、总预算耗尽停止、无 ready 不进入开发、子日志保留轮次和agent。mock 两轮验证流程，真实调用演示单独记录。

下一步可以实现按已保存阶段恢复、更多失败记忆，再考虑 SGPO；不需要先增加新agent。

## 10. 五轮中只升级两次，代码怎样实现

先明确采纳条件，再写替换逻辑。我们沿用模拟 A/B 的规则：最终判断必须为 KEEP，引用已校验、LLM 与规则一致、文字没有被检查出冲突。只有离线 NDCG 提升而 A/B 为 EXTEND 时不升级。这里的采纳是模拟环境下的采纳，不代表真实线上效果。

例如五轮的判决是 KEEP、EXTEND、KEEP、DISCARD、REVIEW_REQUIRED：

| 轮次 | 当轮基线 | 是否采纳 | 下一轮基线 |
| --- | --- | --- | --- |
| 1 | v00 | 是 | v01 |
| 2 | v01 | 否 | v01 |
| 3 | v01 | 是 | v02 |
| 4 | v02 | 否 | v02 |
| 5 | v02 | 否 | v02 |

这个表是测试场景，不是一次真实 DeepSeek 运行的结果。

打开 `workflow/runner.py`：每轮开始从 `state['current_baseline_run']` 获取父模型。评估和反馈成功后检查 KEEP；再调用 `workflow/models.py` 的 `promote`，生成新的 baseline ID，最后更新 current_baseline_run。状态包含 baseline_versions、promotions，各轮包含 baseline_run、adopted、new_baseline_run。主日志明确记录“采纳候选并升级基线”或“保留当前基线”。

### 每种模型保存在哪里

```text
runs/workflow-<id>/
  activity.md
  status.json
  report.json
  models/
    baselines/
      v00/   # 初始基线
      v01/   # 第一次采纳后的基线
      v02/   # 第二次采纳后的基线
    candidates/
      round-01/
      round-02/
      round-03/
      round-04/
      round-05/
```

每个模型目录包含 model.pt（权重）、candidate_model.py（模型源码）、training.toml（训练配置）、report.json（指标和摘要）。候选还包含 decision.json，标明是否采纳、当轮基线和评估位置。训练完成就归档候选，因此评估失败时模型仍可查看；评估未完成时不会有采纳决定。尚未训练成功的轮次，其开发目录与失败日志仍保留。

采纳后的新基线也存放在 runs/baseline-<id>，供已有工具和 A/B 加载。工作流目录中的版本是完整副本，不是依赖原目录的符号链接。report.json 的 promotion 字段记录父基线、候选训练 ID、评估目录和采纳条件。不要手动替换这些文件，否则摘要校验会拒绝使用。

### 为什么修改 runner 还不够

Brainstorm 的 read_file 原来总读项目中的初始 model.py。现在该路径代表当前父模型的代码，返回 actual_source 和 parent_run_id；父模型带有候选快照时实际读取 candidate_model.py。否则它会针对旧代码提出方案。

Developing 原来总复制项目初始代码。现在先读取父报告的 candidate_source，再复制父模型快照，并核对源码和权重摘要。配置来自父报告。因此第二次修改保留第一次已采纳的代码和参数。

实验记忆原来只查同一个父模型。现在沿 report.promotion.parent_run_id 找到当前基线及祖先，再按同一数据版本查询反馈。每条反馈仍保留实际 parent_run_id，不把旧基线的效果当成相对新基线的效果。无关分支的反馈不会进入当前链。

### 如何验证

`tests/test_model_promotion.py` 用五轮确定的评估结果验证两次升级，检查每轮父模型、三版基线、五个候选的完整文件，以及新基线能查询祖先反馈。mock 只用于验证控制流程，不替代模型效果评估。

```bash
mini-agentx iterate --baseline-run baseline-c30ead6da742 --rounds 5
```

最后查看 report.json 的 current_baseline_run 和 promotions，再打开 models/baselines 中已采纳的模型。五轮默认最多 260 次 LLM 请求，遇到失败仍停止；当前不支持断点恢复。

## 批量尝试所有可行提案，重复训练复用历史产物

当前每轮读取 Brainstorm 的完整 proposals.json，按 priority 尝试所有 maturity=ready 的提案。probe_first/backlog 不进入训练。每个候选生成自己的 proposal-NN.json、Developing 目录、候选评估目录及模型归档 models/candidates/round-NN/candidate-NN。一个候选失败记录错误并继续其他候选；如果本轮全部失败，工作流停止；部分失败时显示 completed_with_errors。

所有候选对照本轮开始时的基线，各自评估和反馈入库。本批结束后，从可信 KEEP 候选中选择验证 NDCG 增益最高者升级；未满足 KEEP 的候选不能凭离线提升升级。并列时保留优先级靠前的候选。这是教学选择规则，不等于证明在多重比较下有真实线上收益，模拟 A/B 的限制仍适用。

train_candidate 检查通过后查找同源码、有效配置、种子和数据版本的历史训练。存在时校验权重及候选源码摘要，将权重、源码和报告复制到本次独立训练目录，返回 training_reused_from；不重新训练。摘要错误会拒绝复用。复用的指标仍是原观测，不计为第二次独立验证。A/B 与评估按当前父模型重新执行，不能把旧基线的评估判决照搬。

默认调用预算改为每轮52次（Brainstorm最多10，三个候选各Developing最多12加Evaluation最多2），总上限600次。可以用 --max-model-calls 设置更低预算；达到预算不再发送请求，未完成候选不会被当作成功。旧章节的单候选与每轮24次预算描述属于旧版。主日志末尾的“每个候选的结果”表展示逐个状态，report.json 的 rounds[].candidates 保存所有产物位置与复用来源。
