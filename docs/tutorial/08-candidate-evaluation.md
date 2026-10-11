# 08：把候选模型接到 A/B 和 Evaluation

## 这一章的开发流程与经验

连接两个 agent 时，先定义上一阶段交出的文件，再检查下一阶段实际使用的对象。不要只凭路径或模型声称“完成”就开始评估。

1. 读取开发交接，核对候选文件、训练模型和数据版本。
2. 用实际模型代码加载实际权重，生成 A/B 排名。
3. 把排名交给固定反馈环境，计算观测与护栏。
4. 将去掉规则答案的观测发送给 Evaluation。
5. 保存结论、分歧和失败位置，不直接选用候选。

经验：自定义模型不能用默认模型类代替加载。指标提高、模拟反馈提高和真实业务提高是不同证据。引用数字正确，也不保证模型的文字解释正确。

## 1. 这一阶段要接什么

输入是 Developing 的 handoff.json，输出是模型模式 A/B 报告及 Evaluation 判断。这章没有重新生成提案或训练；完成后自动保存 [第09章反馈](09-experiment-memory.md)，仍不自动晋升。

对应 [AgentX §6](https://arxiv.org/html/2606.26859v2#S6) 的反馈与判断目标；模型加载、文件摘要和本地流程是教程工程选择。我们使用合成反馈替代线上观测，不复现真实部署、流量发布或业务收益。

| 组件 | 输入 → 输出 | 代码 |
| --- | --- | --- |
| 固定交接流程 | 开发目录 → A/B 与评估产物 | [candidate_evaluation.py](../../src/mini_agentx/workflow/candidate_evaluation.py) |
| 模型排名工具 | 训练目录、元信息、训练交互 → 排名 | [isolated_rankings.py](../../src/mini_agentx/abtest/isolated_rankings.py) |
| 子进程 | 模型源码与权重 → 每用户前十未见物品 | [ranking_worker.py](../../src/mini_agentx/abtest/ranking_worker.py) |
| 模拟环境 | 两组排名 → 点击事件与统计 | [model_adapter.py](../../src/mini_agentx/abtest/model_adapter.py)、[simulator.py](../../src/mini_agentx/abtest/simulator.py) |
| 评估 agent | 观测报告 → 判断和引用 | [evaluation.py](../../src/mini_agentx/agents/evaluation.py) |

## 2. 第一步：检查开发交接

打开 run_candidate_evaluation。它要求 status=developed，检查训练目录在这次开发的 training 下，核对 model.py、training.toml 与 handoff.files 的摘要，再检查训练报告的模型摘要和数据版本。

这样，如果你开发后又修改了候选文件，旧交接就不能继续使用。输入不是一个任意源码字符串，模型路径和父实验由已完成的开发产物提供。

父模型 A 来自 runs/parent_run_id；候选 B 来自 handoff.training_run。它们可以位于不同目录，不能只用 B 的末级 run ID 在根 runs 下查找。

## 3. 第二步：让排名工具加载实际模型

### 3.1 工具接收什么

isolated_rankings(run, info, manifest, training) 接收训练目录、训练报告、用户/物品数量和训练交互。它返回 rankings：长度等于用户数，每一项是该用户的前十未见物品 ID。

源码选择规则：报告含 candidate_source 时，只允许 candidate_model.py；普通基线使用仓库固定 model.py。无论哪一种，都要求模型源码与 checkpoint 摘要匹配，然后才启动子进程。

### 3.2 子进程做什么

ranking_worker.py 接收 JSON 文件，检查摘要，动态加载对应 MatrixFactorization，然后加载 checkpoint：

```python
model = module.MatrixFactorization(payload['users'], payload['items'], payload['dimensions'])
model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
model.eval()
with torch.no_grad():
    scores = model.score_all()
```

这是当前代码摘录。检查评分矩阵形状和有限值后，屏蔽训练已见物品，再稳定排序得到前十。score_all 来自实际候选代码，结构改动不会被默认类替代。

### 3.3 主进程还检查什么

子进程输出也需要检查：每用户排名数量正确，物品 ID 是整数且在范围内，没有重复，也没有训练已见物品。运行前后再次核对文件摘要。排名通过后才交给模拟器。

模型代码只在 macOS sandbox-exec 子进程加载，不进入主评估进程。禁网络、清理环境变量、限制写入临时目录、阻止已知凭据路径、限制 CPU 与输出大小、60 秒超时，沿用第 07 章的本机执行边界。文件读取仍较宽、没有完整内存限制，不是可运行恶意代码的完整隔离系统。

## 4. 第三步：运行模型模式 A/B

控制流程固定调用：

```python
ab = simulate(load_ab_config(ab_config), 'null', runs, parent_run, candidate_run)
```

必须是 null 场景，不能人为给候选加 CTR 增益。这里 null 表示不额外注入收益，不代表 A/B 必须无差异。

model_adapter 根据排名选择曝光物品；独立合成物品因子与训练历史偏好生成点击概率。反馈不是 score 的大小，也不使用验证/测试标签生成偏好。因此把分数整体乘大并不会自动制造收益。

事件、稳定分桶、用户级统计与护栏沿用第 04 章。报告 models 保存 A/B 的源码和权重摘要，方便确认比较了哪些产物。模型模式中的延迟和错误仍为合成值，不是实际模型推理性能测量。

## 5. 第四步：Evaluation 每次收到什么

前面的交接、排名和 A/B 全是固定 Python 流程，不调用 LLM，也不添加新工具选择 prompt。Evaluation 的第一次请求仍是：

```text
[0] system：第05章完整判断规则与输出格式
[1] user：evaluation_evidence(A/B report)，递归去掉 rule_decision
```

user 包含模拟标识、policy、analysis、canary、models 的模型身份与摘要、caveats 等观测信息；不读取 simulator_private.json，也不传候选模型原文。当前阶段没有把 Brainstorm 提案的因果解释加入 LLM 证据，只根据 A/B 观测判断。

错误输出需要修复时追加 assistant 原响应与 user 校验错误，仍最多两次尝试。现在每次完整请求保存在 evaluations/<id>/request-NN.json，直接打开即可看到实际 messages。

程序核对引用，再与规则对照。另有一个小范围解释检查：结构化 traffic_balanced=true 时，如果 rationale 明确声称“存在流量不平衡”等，会转 REVIEW_REQUIRED。它只覆盖已知文字冲突，不是完整的自然语言事实验证，未命中也不代表解释完全正确。

## 6. 自己运行

先做已有的 Developing，拿到开发目录 ID。本机已有 developing-da12bb3e161d：

```bash
# 只运行本地模型A/B，不调用LLM
mini-agentx evaluate-candidate --development-run developing-da12bb3e161d --prepare-only

# 模型A/B完成后交给DeepSeek Evaluation
mini-agentx evaluate-candidate --development-run developing-da12bb3e161d
```

每次命令都新建 A/B；prepare-only 完成后若只想评估那一份，使用返回的 ab_run_id：

```bash
mini-agentx ab-evaluate --run <ab_run_id>
```

不要为了得到 KEEP 反复实验。本项目固定窗口的模拟重跑用于验证代码，不提供真实业务显著性证据。LLM失败时，status.json 保留已完成的 ab_run_id，可以对同一份观测重新评估，无需重新生成实验。

## 7. 看日志和真实例子

产物在 runs/candidate-evaluation-<id>，包含 activity.md、activity.jsonl、status.json；成功完成该阶段还保存 handoff.json。阶段状态有 running、ab_completed、evaluated、failed。promoted 始终为 false，evaluated 表示评估流程完成，不代表 KEEP。

A/B 报告、events 和 Evaluation 的 evidence/request/trace/assessment 仍在 runs/ab-<id>。评估失败时不生成成功 handoff，已有 A/B 不丢失。

真实例子：开发候选的离线 NDCG 提高，但模拟 A/B 的 CTR 绝对差约 -0.00228，95% 区间约 [-0.01491, 0.00972]。规则判断 EXTEND：效果证据不足。DeepSeek 也输出 EXTEND，却在解释中说 0.0315 低于 0.001，称流量不平衡，实际 traffic_balanced=true。

我们保留原响应，复核后将这份评估标记 REVIEW_REQUIRED，并加入上述冲突检查。这个例子说明为什么“模型判决与规则相同”仍不足以证明解释可靠。不要把错误解释记入下一轮的已证实经验。

另一次本地测试使用真正改过源码的候选，成功通过隔离加载和模拟 A/B；这与只改 embedding 维度的示例分开验证，不冒充真实 LLM 修改结构的演示。

## 8. 验证与下一步

```bash
python -m unittest discover -s tests -v
```

测试包含非法排名、源码摘要不匹配、开发后文件被改、prepare-only、评估分歧和失败保留 A/B ID，以及引用正确但文字冲突。mock 只验证状态与路由；本地实际隔离加载和真实 DeepSeek 另行运行。

[实验反馈与记忆](09-experiment-memory.md) 已实现，保存方案、修改、指标、判决和限制，供下一轮查询。再连接完整 Brainstorm → Developing → A/B → Evaluation → 记忆循环。当前还没有自动发布、晋升或多轮恢复。
