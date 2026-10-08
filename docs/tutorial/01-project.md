# 01：Python 工程与命令入口

## 论文对应

AgentX §3 描述了多 agent 协作的整体闭环。

本章属于工程支撑：建立统一运行入口，后续通过它启动数据处理、
训练和 agent workflow。Conda、setuptools 和 CLI 是本项目的选择，
不是论文指定的技术。

## 创建环境

在终端执行：

    conda create -n mini-agentx python=3.12 pip -y
    conda activate mini-agentx

之后每次开发，先激活这个环境。

## 项目结构

    pyproject.toml
    src/
      mini_agentx/
        __init__.py
        cli.py
        config.py
    configs/
      baseline.toml

- pyproject.toml：定义包信息、安装方式和终端命令。
- __init__.py：标记 Python 包。
- cli.py：解析命令参数，提供 main() 入口。

## 安装和运行

在项目根目录执行：

    python -m pip install -e .
    mini-agentx --help
    mini-agentx --version

可编辑安装让源码修改直接生效。
修改依赖或命令注册后，需要重新运行安装命令。

## 调用链

    mini-agentx → mini_agentx.cli → main()

pyproject.toml 的 project.scripts 将命令连接到 Python 函数。
入口支持帮助、版本和配置检查，尚未接入 agent。

## 验证结果

已在 mini-agentx Conda 环境中完成可编辑安装。

- --help 正常显示参数说明。
- --version 输出 mini-agentx 0.1.0。

## 下一步

优先学习 Brainstorm Agent 的输入输出契约，再接入 DeepSeek。
真实训练与评估环境在完整闭环前补齐；示例证据不会当成真实实验结果。

## 配置与运行目录

对应论文 §3 的共享数据层目标，本步骤属于工程支撑。
TOML 格式、目录名称和校验规则是本项目的选择。

| 目录 | 用途 | 提交 Git |
| --- | --- | --- |
| configs/ | 可复用的实验配置 | 是 |
| data/raw/ | 原始数据 | 否 |
| data/processed/ | 固定划分和 ID 映射 | 否 |
| runs/ | 实验日志、指标和权重 | 否 |

配置保存在 `configs/baseline.toml`，当前包含非负整数 `seed`
和三个非空目录字符串。配置中的路径约定相对于项目根目录；
当前读取器仅读取和校验字段，不解析路径或创建目录。

`src/mini_agentx/config.py` 使用标准库 `tomllib` 读取 TOML。
文件以二进制模式打开，这是 `tomllib.load()` 的要求。
seed 使用 `type(seed) is int` 检查，避免将布尔值当成整数接受。

在项目根目录执行：

```bash
python -c 'from mini_agentx.config import load_config; print(load_config("configs/baseline.toml"))'
```

预期结果包含 `seed: 42` 和 `paths` 中的三个目录配置。
文件不存在、TOML 格式错误或必需字段无效时，读取器会报错。

## CLI 接入配置

在项目根目录执行：

```bash
mini-agentx check-config --config configs/baseline.toml
```

也可以省略 `--config`，使用默认文件。相对配置文件路径依据当前工作目录解析，因此本章命令统一在项目根目录运行。

调用链：命令 → argparse 子命令解析 → load_config → 字段校验 → JSON 展示。
文件或字段错误转换为简短的终端错误，退出码为 2。
该命令不创建目录、不训练模型，也不调用 LLM。

创建本地运行目录：

```bash
mkdir -p data/raw data/processed runs
```

这些目录被 `.gitignore` 排除，Git 不保存空目录，下载和训练工具后续会负责创建它们。

## 本章逐步骤论文对照

| 步骤 | 论文对应 | 实现位置 | 输入 → 输出 | 差异与验证 |
| --- | --- | --- | --- | --- |
| 环境与安装 | §3，工程支撑 | pyproject.toml | 源码 → 可安装包 | Conda/setuptools 为本项目选择；可编辑安装已成功 |
| 命令入口 | §3，工程支撑 | cli.py | 参数 → 函数调用 | 尚未编排三个 agent；帮助和版本已验证 |
| 配置读取 | §3，工程支撑 | config.py、baseline.toml | TOML → 字典 | 不等于完整共享数据层；seed 和目录字段已读取 |
| 配置子命令 | §3，工程支撑 | cli.py | 文件路径 → JSON 或错误 | 不执行实验；正常与错误路径均检查 |
| 产物目录 | §3，工程支撑 | data/、runs/、.gitignore | 本地文件 → 分类存储 | 尚无知识库或监控平台；忽略规则已检查 |

论文阅读入口：[AgentX §3](https://arxiv.org/html/2606.26859v2#S3)。

## 收尾验证记录

在本机 `mini-agentx` Conda 环境中实际检查：

- 无参数、帮助、版本、子命令帮助均正常退出。
- 默认配置和显式 `--config` 均输出 seed 42 及三个目录。
- 文件缺失、TOML 格式错误、布尔值 seed、空目录字段均以退出码 2 报错，没有 Python traceback。
- 数据与运行产物路径被 Git 忽略，`git diff --check` 通过。

本章完成；没有执行训练或 LLM 调用。
