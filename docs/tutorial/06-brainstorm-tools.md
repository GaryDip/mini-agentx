# 06：Brainstorm 主动调查工具

## 论文对齐

对照 [AgentX §4.3.1–4.3.3](https://arxiv.org/html/2606.26859v2#S4.SS3)：将系统知识、实验知识和数据分析变成可查询工具。首版只实现本地只读查询；§4.3.4 的论文检索、工业 SQL 与动态证据权重仍未实现。

## 使用

```bash
mini-agentx brainstorm --baseline-run <baseline_run_id>
```

现在默认启用调查循环。命令不写候选代码、不训练模型、不运行 A/B。仓库根目录启动，.env 仍只由客户端加载，不开放给工具。

## 五个工具

| 工具 | 输入 | 返回 | 论文对应 |
| --- | --- | --- | --- |
| list_files | 无 | 只读代码/配置白名单 | §4.3.2 |
| read_file | path | 文件内容与摘要 | §4.3.2 |
| list_experiments | 无 | 最多 30 个训练/模型模式 A/B 实验摘要 | §4.3.1 |
| read_experiment | run_id | 验证历史或模型 A/B 观测 | §4.3.1 |
| training_data_summary | 无 | 父模型训练数据的分布统计 | §4.3.3 |

read_file 只允许 model.py、train.py、evaluate.py 及两个指定配置路径，完整白名单见 `tools/brainstorm.py`。读取训练器和评估器是为了理解现有系统，不授权修改。路径穿越、符号链接、任意 shell、.env、测试文件与 simulator_private.json 均不开放。

read_experiment 用固定字段筛选训练报告，不返回测试指标、完整配置或本地数据路径。模型模式 A/B 明确标记 simulated；人为注入收益的场景实验不作为模型优化证据开放。当前目录没有候选失败记忆，不能声称已查询完整历史知识库。

训练统计只读 train.csv，检查父实验数据版本及训练摘要，输出交互数、用户数、物品数、用户交互分布、热门物品占比和稀疏物品数量。不返回原始用户记录，不读取验证/测试标签。

## 工具调用循环

```text
任务及初始证据 → LLM 请求工具 → 程序验证参数并执行
      ↑                              ↓
      └──────── 工具证据返回 ────────┘
                 ↓
            提案、校验、交接
```

这里采用显式 **JSON action 协议**，并非 DeepSeek 原生 function calling：

```json
{"action":"tool","name":"read_file","arguments":{"path":"src/mini_agentx/recommender/model.py"}}
```

程序将请求路由到真实 Python 函数，返回新的证据：

```json
{"tool_result":{"evidence_id":"tool-1","source":"system_kb","status":"observed","facts":{"path":"...","content":"...","sha256":"..."}}}
```

模型可继续请求工具，也可返回 `{"action":"propose","batch":完整候选批次}`。所有工具文本被视为数据而非指令。错误以 tool_error 返回，不能成为事实证据。

为避免跳过调查，必须成功读取模型文件、父实验和训练统计，再允许生成。这个最低步骤是教程的工程约束，不是论文固定规定的工具顺序。其他文件和实验按模型需要查询。选中提案必须至少引用一条本轮新获得的工具证据。

默认每批最多 8 次模型调用、6 次工具调用；最多两次无效输出后停止。工具拒绝也占预算。配置在 `configs/brainstorm-llm.toml` 的 workflow 分组。调用达到预算而未生成有效提案时保存 failed，不伪造成功。

## 如何查看实际工具轨迹

`runs/brainstorm-<id>/tools.json` 记录工具名、参数、成功证据或拒绝原因。
context.json 保存扩充后的证据，trace.json 保存模型请求及提示词。引用由校验器与扩充后的证据核对；工具错误不会进入 evidence。

## 真实验证

本次 `runs/brainstorm-b72b53ff30b0/` 中，DeepSeek 实际调用：

1. read_file：查看模型代码。
2. read_experiment：查看父实验验证历史及训练参数。
3. training_data_summary：查询训练分布。
4. 返回三个候选，选中维度 32→64；引用 tool-3 的真实物品数 1,423。

这是 4 次真实模型调用、3 次工具调用，不是模拟模型响应。候选未训练，不能称为已获得提升。

## 限制与后续

只读白名单是工具接口边界，不等于通用代码执行沙箱；本阶段无写文件或执行 shell 的工具。读取代码使提案有更多依据，但仍不证明机制判断正确或代码可实施。JSON action 与原生函数调用有相同的路由思想，原生调用可作为后续接口升级。

下一步接 Developing，把已经调查过的提案转成候选修改与真实训练；之后接完整闭环和失败记忆。
