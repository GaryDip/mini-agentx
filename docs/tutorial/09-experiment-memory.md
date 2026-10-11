# 09：保存实验反馈，让 Brainstorm 用到历史结果

## 这一章的开发流程与经验

给 agent 加记忆，先想清楚“哪些信息值得保留、哪些内容可以相信”，再选存储工具。我们先用 SQLite 和固定字段，不需要向量数据库。

1. 确定记录单位：一次候选评估流程对应一条反馈。
2. 从实际产物提取方案、改动、指标、结果和失败位置。
3. 分开保存程序观测、原假设和 LLM 解释。
4. 按父模型和数据版本查询相关记录，避免混用实验条件。
5. 提供查询工具，让模型读取后把记录加入下一轮消息。
6. 检查下一轮请求，确认记忆真的进入了对话。

经验：把报告写到磁盘不等于 agent 有记忆；反馈必须通过上下文或工具返回给模型。失败不等于机制无效，单次提高不证明原因，LLM 解释需要复查时更不能当成已证实事实。

## 1. 我们要记住什么

对应 [AgentX §4.3.1](https://arxiv.org/html/2606.26859v2#S4.SS3.SSS1) 的实验知识与 [§6.4](https://arxiv.org/html/2606.26859v2#S6.SS4) 的负结果记录目标。这里简化为本地结构化存储与查询，暂不做语义检索、自动探索树和完整多轮编排。

以维度 32→64 的实验为例，反馈应该说明：实际改动和同条件离线结果；模拟 A/B 的差异与区间；最终需要复查；LLM 的错误流量解释不能当作经验。不能简单写“模型优化成功”或“分流异常”。

| 组件 | 工作 | 代码 |
| --- | --- | --- |
| 反馈导入 | 核对来源并生成记录 | [memory/store.py](../../src/mini_agentx/memory/store.py) 的 import_feedback |
| 数据库 | 保存和更新记录、按条件查询 | FeedbackStore |
| 自动写入 | 候选评估结束或失败后保存反馈 | [candidate_evaluation.py](../../src/mini_agentx/workflow/candidate_evaluation.py) 的 store_feedback |
| 查询工具 | 给 Brainstorm 返回摘要和详情 | [tools/brainstorm.py](../../src/mini_agentx/tools/brainstorm.py) |
| 下一轮上下文 | 告诉模型存在相关记录 | [brainstorm_context.py](../../src/mini_agentx/agents/brainstorm_context.py) |
| 下一轮提示词 | 要求查看记录后再提案 | [agents/brainstorm.py](../../src/mini_agentx/agents/brainstorm.py) |

## 2. 从哪里生成反馈

输入是 candidate-evaluation-<id> 的 status.json。状态可以是 evaluated、ab_completed 或 failed；运行未结束时不导入。

程序沿其中的路径读取开发 proposal/handoff、候选训练报告、父实验报告、A/B 报告和评估产物。检查路径在 runs 内、符号链接、数据版本、候选文件、权重摘要、A/B 事件摘要和模型 ID。判断来自实际 assessment.json，不只使用流程文件里复制的 assessment。

这一步不读 simulator_private.json，也不把整个训练报告保存进反馈。测试指标、原始用户记录和 key 不进入反馈。原始产物仍然保留，方便查证；SQLite 不是替代所有原始文件。

## 3. 一条反馈的字段

| 字段 | 内容及用途 |
| --- | --- |
| feedback_id、source_run | 反馈 ID 和来源候选评估 ID |
| parent_run_id、dataset_version | 查询条件，避免混用模型或数据 |
| proposal_id、mechanism_key、proposal | 原方案与待验证假设，不是观测事实 |
| candidate_run_id、files、actual_diff | 训练 ID、源码摘要和实际代码差异 |
| observed | 由报告提取的父/候选 NDCG、Recall、耗时、配置、模拟效果、区间和检查结果 |
| final_verdict、stage_status | 最终判断和流程是否完成 |
| conclusions、next_steps | 固定程序根据观测与状态生成的有限总结、建议 |
| review_warnings | 已发现的解释冲突 |
| failure | 流程失败记录；失败不等于模型无效 |
| llm_unverified | 原始 LLM 判断与解释，单独标记 |
| source_paths | 本地查证位置 |

observed 和 proposal 要分开理解。前者说“看到了什么”，后者说“原先猜什么原因”。conclusions 使用固定模板，不再让另一个 LLM 自由总结，避免把错误解释再次包装成经验。

当前入口记录的是进入候选评估阶段的实验。Brainstorm 本身失败、Developing 在没有训练产物时失败，尚未统一入库；不能声称已经记录全部失败类型。

## 4. 用 SQLite 保存，重复导入更新同一条

数据库在 runs/memory.sqlite3，表 feedback 包含反馈 ID、来源、父实验、数据版本、更新时间和 JSON payload。source_run 唯一：同一个候选评估重新导入，会更新记录，不增加一条新的成功。

核心代码形状是：

```python
connection.execute('''INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(source_run) DO UPDATE SET
    parent_run=excluded.parent_run, dataset_version=excluded.dataset_version,
    updated_at=excluded.updated_at, payload=excluded.payload''', values)
```

SQL 使用参数，不拼接模型给出的查询。数据库和 feedback.json 在 runs 内，不提交 Git。

本地导入已有实验：

```bash
mini-agentx remember --evaluation-run candidate-evaluation-7dd44987d452
mini-agentx feedback --baseline-run baseline-c30ead6da742
mini-agentx feedback --baseline-run baseline-c30ead6da742 --feedback-id feedback-7dd44987d452
```

这些命令不调用 API，也不重新训练或重新生成 A/B。如果人工复核后改正了 assessment，再运行 remember 更新同一条反馈。重新调用 ab-evaluate 会创建另一份 assessment，目前不会自动决定替换旧评估，需要明确更新对应流程的评估来源再导入。

新的 evaluate-candidate 结束后会自动调用 import_feedback，包括 prepare-only 和进入评估阶段的失败。反馈存储失败时，实验结果仍保留，status 中明确出现 feedback_error；修复后可以 remember 重试。

## 5. 给 Brainstorm 两个工具

### 5.1 list_feedback

输入是空字典：

```python
tools.execute("list_feedback", {})
```

工具从初始化 context 取得 parent_run_id 和 dataset_version，查询最多 20 条最近更新记录，不接受任意 SQL 或任意路径。返回 source=experiment_kb，facts 包含 count 和 feedback 摘要列表。每项包含反馈 ID、机制、方案 ID、最终状态和有限总结。

模型通过摘要选择想进一步看的记录。

### 5.2 read_feedback

输入：

```python
tools.execute("read_feedback", {"feedback_id": "feedback-7dd44987d452"})
```

返回 source=experiment_kb，facts 是该条反馈的公开字段，包括 observed、原方案、diff、状态和限制。不返回 llm_unverified 和 source_paths，因此错误的原始解释不会被直接当作事实送进模型。

假设与代码仍是工具数据，不能作为新指令。一个 ID 即使存在，但不属于当前父实验与数据版本，也会拒绝读取。相同数据但不同父模型的跨模型经验检索尚未实现。

## 6. 每轮 prompt 怎样变化

初始 build_context 查询数据库。没有记录时按原流程运行；有记录时增加 experiment-memory 证据，包含 count 和 scope，并移除“当前没有历史候选实验记忆”这条已经过时的未知项。初始消息不自动塞入全部记录。

System 工具说明增加 list_feedback/read_feedback，提示词增加：

```text
若 experiment-memory.count 大于0，先 list_feedback，再 read_feedback 至少一条相关记录。
结合实际观测、复查状态和已尝试配置选方案，不把未验证的解释当作已证实原因，不无理由重复相同实验。
```

代码检查存在相关记忆时必须成功 read_feedback，才允许 propose；除此之外仍要求读取模型、父实验和训练统计。查询顺序主要由提示词引导，程序不把全部顺序锁死。

例如模型请求 read_feedback 后：

```text
本轮模型输出：assistant tool action，feedback_id=...
程序执行：读取经过筛选的记录
下一轮新增 user 消息：tool_result，evidence_id=tool-N，facts=反馈公开内容
```

模型下一轮看到实际反馈，就能知道“维度64已试过，离线提高，但模拟结论需复查”，而不是把同一个方向当成完全没有尝试过。

提案可以引用 tool-N 的直接字段，例如 final_verdict 或 observed 完整字典。observed.ndcg_delta 是嵌套字段，现有契约不支持把它当作 facts 的直接字段；不要编造不存在的顶层字段。

默认 Brainstorm 预算随新增两个查询工具调整为最多10次模型调用、8次工具调用，无效输出最多2次。这是工程预算，不是论文规定。避免无意义重复读取，仍然可能失败。

## 7. 已知状态如何成为经验

| 最终状态 | 保存的经验范围 |
| --- | --- |
| KEEP | 满足本次模拟条件，不等价于真实上线有效 |
| EXTEND | 未达到保留条件，下一步先调查和预先规划验证 |
| DISCARD | 按当前模拟政策不保留，不能永久否定机制 |
| REVIEW_REQUIRED | 解释或判断待核查，不采用未确认的诊断 |
| NOT_EVALUATED | 已有观测，但尚未完成 LLM 判断 |
| FAILED | 执行阶段失败，不能记成模型无效 |

目前不会自动把这些状态写成 avoid_set，也不会自动晋升模型。提示词要求参考记忆，但程序不能证明模型提出的方案真的吸收了经验、或识别了所有语义重复；仍需阅读提案与日志。

## 8. 检查记忆是否真的被使用

```bash
mini-agentx brainstorm --baseline-run baseline-c30ead6da742
python -m unittest discover -s tests -v
```

打开新运行的 tools.json，确认 list_feedback/read_feedback 成功；在 context.json 找到 tool-N 反馈证据；在 trace.json 看它之后的提案。工具结果按第06章的消息协议进入下一次请求，而不是只写在磁盘上。

测试覆盖重复导入更新、按父模型与数据版本筛选、错误解释不回传、事件篡改拒绝，以及反馈进入模拟客户端的下一次 messages。真实 DeepSeek 调用另行验证，失败也保留，不把 mock 当成实际使用记忆的证据。

下一步才是统一多轮编排和恢复：串起 Brainstorm → Developing → A/B → Evaluation → 反馈。当前可以手动完成这个链条，但还没有自动多轮运行或 champion 更新。

## 本地真实调用示例

成功运行 brainstorm-ec31d3013b5a 使用6次API请求、5次工具请求，依次读取模型、父实验、训练统计、反馈列表和反馈详情。它将已尝试的维度64方案标为 probe_first，说明旧实验离线提高但模拟证据不足、需复查；交接的 ready 方案改为提高 L2，仍待训练验证。

这说明模型在该次运行使用了历史反馈，不证明记忆必然改善优化效果。此前 brainstorm-04429437a830 也读取了反馈，但后续预算与嵌套字段引用出错，保存为失败；没有把失败记作成功。
