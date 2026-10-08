# 03：推荐基线与可信离线评估

## 论文对应

[AgentX §5.2.1](https://arxiv.org/html/2606.26859v2#S5.SS2) 的模型实验流程在本项目简化为本地 CPU 训练。BPR-MF 是教学替代，非论文 RankMixer。本章的训练与离线评估一起，为 Brainstorm 提供真实证据；模拟 A/B 是独立反馈环境，不把离线指标称为线上收益。

## 模型与损失

用户和电影各有一个 32 维 embedding，点积为分数。对用户 u、正例 i 和训练中未见的采样电影 j，损失为 `softplus(score(u,j)-score(u,i))`，即 BPR 排序损失。加入按批次平均的三组 embedding 平方和正则。

负采样只依赖训练正例，不参考验证/测试标签；未交互物品不代表明确不喜欢。首版预生成每用户可采样物品列表，适合小数据集，不能直接推广到工业规模。

## 运行

在项目根目录、mini-agentx Conda 环境运行：

```bash
python -m pip install -e .
mini-agentx train --config configs/baseline.toml
mini-agentx evaluate --run <训练输出的 run_id> --split validation
```

默认 CPU、单线程、seed 42、Adam 0.001、batch 1024、20 轮、L2 系数 0.0001。每轮使用一次训练交互和一个负例。使用验证 NDCG@10 选最佳 checkpoint，分数相同保留较早一轮。

## 产物

`runs/<run_id>/model.pt` 保存最佳权重，`report.json` 包括配置、数据版本、各轮 loss/验证指标、热门推荐结果、训练耗时、PyTorch 版本和权重摘要。评估命令另写 validation_metrics.json。

报告暂含本地配置与路径，构建 agent 上下文时只选取必要信息。本章不调用 DeepSeek，不读取 .env，不评估测试集。

## 逐步骤论文对照

| 实现 | 对应与类型 | 代码位置 | 输入 → 输出 | 保留/简化 |
| --- | --- | --- | --- | --- |
| 候选模型 | §5.2.1；简化 | recommender/model.py | ID → 分数 | 小模型替代工业骨架 |
| 训练 | §5.2.1；简化 | recommender/train.py | 固定训练集 → 权重 | 本地训练替代平台任务 |
| 记录与 checkpoint | §3、§5.2.1；工程支撑 | report.json | 轨迹 → 可追溯产物 | 单实验文件，未接 agent |
| 参照比较 | §5.2.2；简化 | 热门推荐 | 训练频数 → 验证指标 | 检查基本有效性，非完整消融归因 |

训练产物为 [Brainstorm](06-brainstorm.md) 提供证据；[模拟 A/B](04-ab-environment.md) 和 [Evaluation Agent](05-evaluation-agent.md) 提供另一类反馈。候选修改和完整闭环尚未实现。

## 实际运行结果

本机 Python 3.12、PyTorch 2.14.1、CPU 单线程，seed 42；20 轮最佳 checkpoint 为第 20 轮。

| 模型 | 验证 NDCG@10 | 验证 Recall@10 |
| --- | --- | --- |
| 热门推荐 | 0.03191654 | 0.06310160 |
| BPR-MF | 0.04143243 | 0.08770053 |

935 个验证用户、1,423 个候选电影。两次训练耗时约 6.02 和 6.70 秒（不含首次导入），逐轮 loss、指标及权重张量完全一致。保存权重重新评估得到相同指标。测试集未评估；这是单种子的基线检查，不是统计显著性证明。

本机首次运行目录为 `runs/baseline-c30ead6da742/`，可使用该 run ID 评估验证集。其他机器运行会生成新的 ID。运行产物不提交 Git。

## 离线评估的完整实现

### 论文对应

对应 [AgentX §5.2.2 与 §6](https://arxiv.org/html/2606.26859v2#S6) 的客观反馈目标。这里使用固定离线指标作教学实现，不复现线上 A/B、统计显著性或业务护栏；模拟 A/B 环境已在 [下一章](04-ab-environment.md) 实现。

### 指标协议

训练词表中的所有电影作为候选。验证屏蔽训练已见物品；最终测试屏蔽训练与验证已见物品。每用户只有一个有效留出正例：Recall@10 等于前十命中率；命中排名 r 从 1 开始时 NDCG@10 为 `1/log2(r+1)`，未命中为 0。对对应划分的可评估用户平均，报告用户数和候选物品数。相同分数按电影 ID 升序打破平局。

热门推荐的分数只来自训练交互计数，也屏蔽已见电影。加载数据时检查文件摘要，加载模型时核对数据版本。无有效用户或分数非有限时拒绝评估。

### 验证与运行

```bash
python -m unittest discover -s tests -v
mini-agentx evaluate --run <run_id> --split validation
```

指标测试包含手算排名、已见物品屏蔽、平分和 NaN。测试评估入口已提供，但仅在最终选定模型后手动使用：本次基线训练不运行测试评估，也不将测试信息交给 LLM。

本章暂未定义候选晋升阈值、schema 版本或决策规则；这些将在候选实验与模拟 A/B 接入时确定，不能视为完整 Evaluation Agent。

### 实际检查记录

五项测试通过：手算指标与已见屏蔽、平分/未命中、NaN 拒绝、BPR 单步提高正例相对分数、数据预处理契约。真实基线加载权重后验证指标与训练报告一致。训练和验证流程不加载测试标签（完整性检查只核对文件摘要）。