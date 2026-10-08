# 06：动手实现 Brainstorm Agent

这章按写代码的顺序学习已有实现：先定义输入，再写一个工具，接上模型调用循环，最后校验提案并保存日志。代码已经在仓库里，你不需要重新复制一套；请打开对应文件，把每段代码与下面的解释对照起来。文中的练习可以直接在项目根目录运行，前面的练习不需要 API 密钥。

完成后，你应当能够自己给 agent 增加一个只读工具，并解释它如何从“模型请求”变成“真实执行”，以及工具结果如何影响下一次请求。

## 开始前：先认清我们要写什么

对照 [AgentX §4](https://arxiv.org/html/2606.26859v2#S4)：任务边界对应 §4.1，小批候选对应 §4.2，系统/实验/数据证据对应 §4.3，提案交接对应 §4.4。这里用本地文件和 Python 函数做简化实现。JSON action 协议、必读检查与固定次数预算是教程的工程选择，论文未规定这些具体写法。

先打开这些文件，按顺序阅读：

1. [brainstorm-task.json](../../configs/brainstorm-task.json)：这次优化允许做什么。
2. [brainstorm_context.py](../../src/mini_agentx/agents/brainstorm_context.py)：已有事实从哪里来。
3. [tools/brainstorm.py](../../src/mini_agentx/tools/brainstorm.py)：程序可以执行哪些调查。
4. [agents/brainstorm.py](../../src/mini_agentx/agents/brainstorm.py)：模型与工具如何反复交互。
5. [brainstorm_contracts.py](../../src/mini_agentx/agents/brainstorm_contracts.py)：什么样的方案能进入下一阶段。

不要从整段 prompt 开始背字段。我们先做出一次真实工具调用，再接上循环。整个程序最终应该完成：

```text
任务 + 初始证据 → 模型请求调查 → Python执行 → 观测反馈
                         ↑                    ↓
                         └──── 下一次请求 ────┘
                                  ↓
                          提案 → 校验 → 交接
```

## 第 1 步：用数据结构固定任务边界

**要实现什么：** 把“优化推荐系统”变成程序能够检查的输入。否则模型可能为了提高指标改评估器，也可能提出当前训练器不支持的方案。

打开 brainstorm-task.json，先找 objective、primary_metric、allowed_files、constraints。当前允许修改候选 model.py 的表示与打分，以及 training.toml 的已有训练参数；BPR 损失、负采样算法、数据划分与评估器固定。候选目录尚未由 Developing 创建，Brainstorm 只写计划。

再打开 brainstorm_contracts.py，看 `TaskBoundary.from_dict` 的入口：

```python
require_fields(value, cls.__dataclass_fields__, "TaskBoundary")
if type(value["schema_version"]) is not int or value["schema_version"] != 1:
    raise ValueError("不支持的任务版本")
for name in ("task_id", "objective", "primary_metric"):
    text(value[name], name)
if value["primary_metric"] != "ndcg@10":
    raise ValueError("首版固定使用验证 ndcg@10")
```

`require_fields` 拒绝缺失或额外字段。`type(...) is int` 还避免把 Python 的 True 当成整数 1。`primary_metric` 的检查让后续候选在相同口径上比较。dataclass 负责字段结构，from_dict 才负责运行时检查；只写类型标注并不能阻止错误 JSON。

**动手检查：** 在终端执行下面整块命令。这里的 Python 放在 heredoc 中，不是把文件内容逐行当 shell 命令执行。

```bash
python - <<'PYCODE'
import json
from pathlib import Path
from mini_agentx.agents.brainstorm_contracts import TaskBoundary
value = json.loads(Path("configs/brainstorm-task.json").read_text())
print(TaskBoundary.from_dict(value).allowed_files)
value["primary_metric"] = "ctr"
try:
    TaskBoundary.from_dict(value)
except ValueError as error:
    print("预期拒绝：", error)
PYCODE
```

先看到允许修改的文件，再看到指标被拒绝。这个练习只改内存中的字典，不修改你的配置。

## 第 2 步：给模型准备初始上下文

**要实现什么：** 从父实验提取事实，避免直接把整个报告扔给模型。对应论文 §4.3 的证据输入。

打开 brainstorm_context.py 的 build_context，先看读取与完整性检查：

```python
task = json.loads(Path(task_path).read_text())
TaskBoundary.from_dict(task)
baseline = Path(baseline)
report = json.loads((baseline / "report.json").read_text())
checkpoint_hash = hashlib.sha256((baseline / "model.pt").read_bytes()).hexdigest()
if checkpoint_hash != report["checkpoint_sha256"]:
    raise ValueError("基线权重摘要不匹配")
```

task 的结构先被校验，再检查 model.pt 与报告是否匹配。接下来的 context 字典保留 parent_run_id 和 dataset_version，让后续提案知道自己基于哪个模型、哪一版数据。

重点阅读 `evidence` 列表，而不是只关注指标数值。每条证据有固定 ID、source、status、facts。system-model 是 documented 系统说明；baseline-validation 和 popularity-validation 是 observed 实验观测。模型提出的原因解释应该写入 hypothesis，不能混进 observed 事实。

**动手检查：** 先运行 `mini-agentx brainstorm-context --baseline-run <你的基线ID>`，然后打开命令输出的 context_path。找到 baseline-validation 的 ndcg@10，再对照父实验 report.json。确认值来自真实报告，并找出为什么没有测试指标。这个命令不会调用 LLM。

## 第 3 步：先写一个不依赖 LLM 的工具

**要实现什么：** 提供一个输入明确、结果明确、可以单独测试的 Python 函数。我们先从 read_file 入手，对应论文 §4.3.2 的系统调查。

打开 tools/brainstorm.py。`SPECS` 是给模型看的说明书，`FILES` 是程序允许访问的文件。`execute` 才执行工具：

```python
def execute(self, name, arguments):
    if name not in self.SPECS:
        raise ValueError("未知或禁止的工具")
    if not isinstance(arguments, dict) or set(arguments) != set(self.SPECS[name]["arguments"]):
        raise ValueError("工具参数字段不符合定义")
    if name == "list_files":
        return "system_kb", {"paths": list(self.FILES)}
    if name == "read_file":
        path = arguments["path"]
        if not isinstance(path, str) or path not in self.FILES:
            raise ValueError("文件不在只读白名单")
        target = self.safe_file(self.root, path)
        if target.stat().st_size > 24000:
            raise ValueError("文件超过读取大小上限")
        return "system_kb", {"path": path, "content": target.read_text(),
                              "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
```

顺着分支逐行看：

1. 工具名必须在 SPECS 中，模型不能凭空发明 shell 工具。
2. arguments 必须是字典，并且字段集合与定义完全一致。
3. read_file 的 path 必须在 FILES 中；read 权限与候选 write 权限分开。
4. safe_file 拒绝符号链接并检查解析后的路径范围。
5. 限制文件大小后返回 `(source, facts)`，其中 content 是实际文件内容，sha256 用于标识读到的版本。

**动手调用真实工具：**

```bash
python - <<'PYCODE'
from pathlib import Path
from mini_agentx.tools.brainstorm import BrainstormTools
root = Path.cwd()
tools = BrainstormTools(root, root / "runs", {"parent_run_id": "unused"})
source, facts = tools.execute("read_file", {
    "path": "src/mini_agentx/recommender/model.py"
})
print(source, facts["path"])
print(facts["content"])
try:
    tools.execute("read_file", {"path": ".env"})
except ValueError as error:
    print("预期拒绝：", error)
PYCODE
```

这个练习无需父实验或 API：read_file 不使用 parent_run_id。你已经执行了一次 agent 的真实工具，只是工具名由你指定。下一步让模型选择工具名。

另外四个工具也用同一个 execute 入口：list_files 返回可读路径；list_experiments 查本地实验摘要；read_experiment 筛选验证历史或模型模式模拟 A/B；training_data_summary 检查训练数据摘要后计算分布。实验查询不开放注入收益的模拟场景，训练统计不读取留出标签。

## 第 4 步：让模型输出“动作”，而不是一段建议

**要实现什么：** 给程序一个可以路由的响应协议。打开 agents/brainstorm.py 的 tool_prompt，找到这两种格式：

```json
{"action":"tool","name":"read_file","arguments":{"path":"src/mini_agentx/recommender/model.py"},"reason":"确认模型接口与打分实现。"}
```

```text
{"action":"propose","batch":完整候选批次,"reason":"简短的选择依据"}
```

上面是协议示例，不是一次真实响应。action=tool 表示继续调查，action=propose 表示结束调查并提交批次。reason 请求一两句公开的行动目的，便于阅读日志；它不是内部完整思维链，也不证明方案有效。

我们使用普通 JSON 输出，不是原生 function calling。因此工具不会因为模型输出名字就自动执行；接下来必须自己写 action 检查与 Python 路由。

**阅读客户端：** 打开 llm/client.py 的 complete。messages 被发送到 chat/completions，response_format 请求 JSON，响应 content 被 json.loads 解析。complete 一次只发一次请求，返回 `(output, metadata)`。解析成功只说明输出是 JSON，不能保证工具名合法或证据正确。

## 第 5 步：创建对话状态，启动循环

**要实现什么：** 保留上一轮观测，让下一次请求知道刚刚发生了什么。打开 generate_proposals，找到 messages 与 inspected 初始化：

```python
messages = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)}]
attempts = []
tool_history = []
inspected = set()
```

system_prompt 定义职责、约束、协议和工具说明；第一条 user 消息携带 task/context。messages 是模型每次请求的对话历史，context 是程序保存的事实集合，tool_history 是工具执行记录，inspected 是最低调查完成状态。四者用途不同。

循环调用模型的部分是：

```python
for index in range(max_calls):
    activity.record(f"调用模型：第 {index + 1} 次")
    try:
        candidate, metadata = client.complete(messages)
    except LLMError as error:
        attempts.append({"attempt": index + 1, "status": "api_failed", "error": str(error)})
        activity.record("模型调用失败", result={"error": str(error)})
        if "HTTP 4" in str(error) and "HTTP 429" not in str(error):
            break
        continue
```

网络或 API 错误会被记录。不可恢复 4xx 停止，429 等错误可在剩余总次数内继续。这里没有“无限自动重试”，每个循环都消耗总模型调用预算。

## 第 6 步：执行工具，把结果变成可引用证据

**要实现什么：** 将模型请求路由到第 3 步的真实工具，并把结果放进 context。先读 action 类型、字段和工具预算检查，再读这段成功分支：

```python
source, facts = tools.execute(candidate["name"], candidate["arguments"])
evidence = {"evidence_id": f"tool-{len(tool_history) + 1}",
            "source": source, "status": "observed", "facts": facts}
context["evidence"].append(evidence)
entry.update(status="success", evidence=evidence)
```

工具返回 source/facts，编排器添加 evidence_id 与 observed 状态，得到统一证据格式。context.evidence.append 让后面的提案校验器能够找到这条证据。entry 保存同一次执行结果，用于日志。

tool-N 的 N 按工具请求次数生成，失败请求也计数，所以成功证据 ID 可能不连续。不要用 ID 的连续性判断有没有丢数据。

继续往下看 inspected：成功读到 model.py 标记 model，成功读父实验标记 parent，成功查询训练统计标记 training。提交提案前要求这三项都完成；执行顺序不由集合固定。这个必读检查是本地教程的工程约束。

工具异常产生 tool_error，记录拒绝但不添加 observed 证据。看完这里，你应能回答：为什么读取 .env 被拒绝后，模型不会得到密钥，也不能把错误当成事实引用？

## 第 7 步：把工具结果送回模型——循环最关键的一步

**要实现什么：** 让模型真正看到执行结果，而不是仅在磁盘上保存它。继续读同一函数：

```python
tool_history.append(entry)
activity.record(f"工具结果：{candidate['name']}", result=entry.get("evidence", {"error": entry.get("error")}))
record["status"] = "tool_action"
attempts.append(record)
messages.extend([
    {"role": "assistant", "content": json.dumps(candidate, ensure_ascii=False)},
    {"role": "user", "content": json.dumps(observation, ensure_ascii=False)},
])
continue
```

这段代码有两个不同的效果：activity 和 attempts 保存给人看的执行记录；messages.extend 把请求与观测交给下一次模型调用。只写日志、不追加 messages，模型就不知道工具实际返回了什么。

assistant 消息保存模型自己的 action；user 消息包含 Python 返回的 tool_result 或 tool_error。因为采用普通 JSON 协议，这里没有使用原生 tool role。continue 回到循环头，下一次 complete 接收到扩展后的对话。

**跟踪一次状态：** 对照 tests/test_brainstorm_tools.py 的 `test_investigation_results_become_citable_evidence`。FakeClient 顺序返回三个 tool action 与一个 propose。第三个工具返回 users=2，最终提案引用 tool-3.users=2。这个测试让你在不用 API 的情况下看到“请求 → 结果 → 可引用证据”完整链路。

```bash
python -m unittest discover -s tests -p 'test_brainstorm_tools.py' -v
```

这里的 StubTools 和 FakeClient 是教学测试替身，不代表真实文件或真实 DeepSeek。第 3 步已经单独调用过真实文件工具，两种验证解决不同问题。

## 第 8 步：接住提案，检查它能不能交接

**要实现什么：** 从合法 JSON 走到可交接方案。打开 brainstorm_contracts.py 的 Proposal.from_dict，重点看引用检查：

```python
for reference in references:
    require_fields(reference, ("evidence_id", "field", "value"), "evidence_ref")
    if not isinstance(reference["evidence_id"], str) or reference["evidence_id"] not in evidence:
        raise ValueError("引用了不存在的 evidence_id")
    field = text(reference["field"], "field")
    actual = evidence[reference["evidence_id"]]["facts"].get(field)
    if field not in evidence[reference["evidence_id"]]["facts"] or not same_value(reference["value"], actual):
        raise ValueError(f"证据字段或数值不匹配：{reference['evidence_id']}.{field}；必须使用原始 JSON 值 {actual!r}（{type(actual).__name__}），不能将数值转成字符串")
```

校验器按 evidence_id 找证据，再从 facts 的直接字段取值。32 与 "32" 不能混用；未知 ID、字段或错误值都会被拒绝。它证明引用一致，不证明这条事实支持因果假设。

然后阅读 changes、validation 和 maturity 检查：ready 需要修改计划且不能有阻塞 probes；probe_first 必须说明先调查什么；backlog 表示当前条件不足。validation 固定主指标与同数据同种子的对照方法，还要求 observables 与 falsification。

最后 validate_batch 选择可交接提案：
```python
ready = sorted((item for item in proposals if item.maturity == "ready"), key=lambda item: item.priority)
```

这是现有选择语句：数字越小优先级越高，取第一个 ready；没有 ready 则 needs_evidence。上面的数量、任务/父实验匹配和重复机制检查针对整批执行，一个候选不合格就整批拒绝。

回到 generate_proposals，你还会看到选中提案必须引用本轮成功调查证据的检查。它避免“工具都读了，但方案完全忽略新观测”。ready 仍只是通过契约，Developing 还需要判断计划是否能在真实代码中实施。

**动手制造一次校验错误：**

```bash
python - <<'PYCODE'
import json
from pathlib import Path
from mini_agentx.agents.brainstorm_contracts import validate_batch
base = Path("examples/brainstorm")
context = json.loads((base / "context.json").read_text())
batch = json.loads((base / "proposals.json").read_text())
print("原始交接：", validate_batch(batch, context)["handoff_status"])
batch["proposals"][0]["changes"][0]["file"] = "../evaluate.py"
try:
    validate_batch(batch, context)
except ValueError as error:
    print("预期拒绝：", error)
PYCODE
```

示例数据是 synthetic_example。练习只修改内存，不改评估器或示例文件。

## 第 9 步：给模型一次修复机会，设置停止条件

**要实现什么：** 把明确错误反馈给模型，同时避免无限循环。读 generate_proposals 的校验异常分支：

```python
except ValueError as error:
    record.update(status="invalid_output", error=str(error))
    activity.record("程序校验失败", result={"error": str(error), "response": record["response"]})
    attempts.append(record)
    invalid_outputs += 1
    if invalid_outputs >= client.config["max_attempts"]:
        break
    messages.extend([
        {"role": "assistant", "content": json.dumps(record["response"], ensure_ascii=False)},
        {"role": "user", "content": f"校验失败：{error}。按原始任务及证据修复。" +
         ("保持 action 协议，提案用 propose 包装完整 batch。" if tools is not None else "返回完整 JSON 批次。")},
    ])
    continue
```

错误响应以 assistant 消息保存，具体校验错误以 user 消息追加；下一轮仍基于同一个任务与原证据修复。invalid_outputs 累计到 max_attempts 后停止，不是每次工具再重试两遍。

打开 configs/brainstorm-llm.toml：总模型请求最多 8 次，工具请求最多 6 次，无效输出最多 2 次，每次输出最多 4,500 token，网络超时 45 秒。被拒绝的实际工具请求也占工具次数，API 错误也占模型次数。

```bash
python -m unittest discover -s tests -p 'test_brainstorm_agent.py' -v
```

读 `test_repair_then_persist_handoff`：第一次路径越界，第二次返回合法批次；断言 trace 状态从 invalid_output 变为 valid。再读 budget_failure：所有尝试失败时保存 failed，没有成功 handoff。

## 第 10 步：保存日志与下游输入

**要实现什么：** 让人可以复查整个过程，让 Developing 可以继续处理。打开 activity_log.py 的 record，它每次追加 activity.jsonl 和 activity.md，而不是等结束才一次性写日志。因此运行中就能看到请求、公开理由、结果和校验。

接着读 generate_proposals 最后保存产物的部分：context.json 保存调查后的证据；tools.json 保存执行记录；trace.json 保存 prompt、响应、usage 与错误。通过校验才保存 proposals.json 和 handoff.json；只有选中 ready 才写 proposal.json，包含任务、父实验、数据版本与选中方案。

activity.md 是阅读入口，proposal.json 是下一阶段输入。API 密钥不会进入这些文件，内部思维链不记录。当前统一日志只接到 Brainstorm，Evaluation 仍用自己的 evidence/trace/assessment 文件。

## 第 11 步：用 CLI 把刚才的组件接起来

打开 cli.py 的 brainstorm 分支，看组合代码：
```python
context = json.loads(Path(context_path).read_text())
tools = BrainstormTools(Path.cwd(), runs, context)
result = generate_proposals(context_path, runs, DeepSeekClient(args.llm_config), tools=tools)
```

这三行摘自实际 CLI。之前的分支根据 --baseline-run 调用 build_context，或者接受 --context；然后构造客户端与工具，交给 generate_proposals。agent 不需要框架才能工作，核心是明确的接口和有限循环。

配置好 .env 后，在项目根目录执行：

```bash
conda activate mini-agentx
mini-agentx brainstorm --baseline-run <你的基线run_id>
```

本机已有 baseline-c30ead6da742，其他机器用自己的 ID。默认任务文件是 brainstorm-task.json，LLM 配置是 brainstorm-llm.toml，可通过 --task、--llm-config 覆盖。该命令会消耗 DeepSeek API，不会修改模型、训练候选或启动 A/B。

打开输出的 activity_log，选一次 read_file，沿着请求参数、返回的 tool-N、最终 evidence_refs 追踪。再对照源码，看自己能否指出 messages.extend、context.evidence.append、validate_batch 各负责哪一段。

## 第 12 步：练习给 agent 增加一个工具

理解调用循环后，尝试在本地增加只读工具 `model_parameter_count`，统计父模型配置对应的参数量。当前 BPR-MF 无偏置，参数量可由 `(users + items) * dimensions` 计算，但必须从可信父实验/数据元信息取得输入，不能用模型自己编造的数量。

这是练习要求，当前仓库**尚未提供这个工具**。按以下顺序动手：

1. 在 BrainstormTools.SPECS 增加名字、空 arguments 与说明。
2. 在 execute 增加分支，复用安全路径读取与完整性检查，返回 source/facts，包含参数量、维度、模型类型及数据版本。
3. 加一个独立工具测试，校验返回值，并检查不会读取测试标签或密钥。
4. 在调用循环测试中插入该 action，并让提案引用新证据；检查消息回传与引用校验。
5. 最后才用真实 API 验证模型是否会根据需要选择它。它不属于必读项，不必修改 inspected。

如果工具采用现有 `(source, facts)` 返回协议，就不需要在 generate_proposals 再加同名分支：通用 execute 路由已经负责调用。这是把工具与编排分开的价值。

## 接下来要写的代码

Developing 输入 proposal.json，计划提供候选文件读写、接口检查、固定训练入口和诊断工具，复用同一种请求/执行/反馈循环。它尚未实现；执行隔离、训练超时和有限修复要在这一阶段落实。完整实验记忆、外部论文检索、动态证据权重和多批探索也仍在后续范围。

继续学习 [Evaluation 教程](05-evaluation-agent.md)，比较“主动工具调查”与“固定证据分析”两种流程；再按 PLAN.md 进入 Developing。
