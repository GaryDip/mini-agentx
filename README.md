# Mini AgentX

一步一步实现推荐系统自动优化 agent 的中文学习项目，依据 [AgentX 原论文](https://arxiv.org/abs/2606.26859) 的机制设计。非官方实现。

目标闭环：**提出假设 → 修改代码 → 真实训练 → 评估结果 → 实验记忆 → 下一轮**。

使用 Python、PyTorch、MovieLens 100K 与 DeepSeek API。先用显式工作流理解上下文、工具调用、状态和记忆，再扩展故障恢复与提示词优化。

## 从哪里开始

阅读 [教程路线图 PLAN.md](PLAN.md)，每章包含目标、实现步骤、方法和验收条件。

教程 01 已完成：Python 包结构、命令入口、配置读取和运行目录规范。已实现 BPR-MF 训练和离线评估；模拟 A/B 已接入 DeepSeek Evaluation Agent；Brainstorm 已接入 DeepSeek 只读调查、生成与校验；Developing 已实现候选修改、隔离检查与真实训练；候选 A/B 与 Evaluation 交接已实现；实验反馈存储与 Brainstorm 查询已实现；自动多轮闭环尚未实现。

## 安装与运行

在项目根目录执行：

```bash
conda create -n mini-agentx python=3.12 pip -y
conda activate mini-agentx
python -m pip install -e .
mini-agentx --help
mini-agentx --version
mini-agentx check-config --config configs/baseline.toml
```

如果已经创建环境，只需激活环境，无需重复创建。

## 教程进度

1. [Python 工程与命令入口](docs/tutorial/01-project.md)
2. [数据环境](docs/tutorial/02-data.md)：下载、检查、固定划分与 ID 映射。
3. [推荐基线与可信离线评估](docs/tutorial/03-baseline.md)：BPR-MF、训练、指标协议与验证。
4. [模拟 A/B 环境](docs/tutorial/04-ab-environment.md)：稳定分桶、合成反馈、用户级统计与护栏。
5. [Evaluation Agent](docs/tutorial/05-evaluation-agent.md)：从 A/B 证据到 LLM 判断、事实校验与决策。
6. [Brainstorm Agent 完整教程](docs/tutorial/06-brainstorm.md)：论文对齐、任务与提案契约、工具定义、JSON 调用循环、预算、交接及日志。

7. [Developing Agent](docs/tutorial/07-developing.md)：候选文件工具、隔离检查与训练、逐轮提示词和开发交接。

8. [候选 A/B 与 Evaluation 交接](docs/tutorial/08-candidate-evaluation.md)：隔离加载实际模型、模拟反馈、请求日志与分歧处理。

9. [实验反馈与记忆](docs/tutorial/09-experiment-memory.md)：SQLite、观测与解释分离、反馈工具和下一轮消息。

教程每章只保留一个 Markdown 文件，旧的拆分与跳转文件已删除。优先学习 agent 时，先阅读 06，再阅读 05；Developing 见第 07 章；后续闭环的实施步骤见 PLAN.md，尚未实现。

## 复现范围

- Brainstorm、Developing、Evaluation 三个 agent。
- 固定数据划分、推荐基线、可信离线评估。
- 实际代码修改、模型训练、实验记忆和多轮迭代。
- 后续简化 SGPO 和配对回放。

离线反馈用于学习闭环，不等价于生产线上 A/B。本项目不声称复现论文的工业规模或业务收益。

## 仓库内容

提交源代码、配置模板、教程和脱敏示例。数据、权重、运行日志、数据库及 API 密钥不入 Git。密钥通过环境变量读取，请勿提交 `.env`。
