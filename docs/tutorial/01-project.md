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
当前入口只支持帮助和版本输出，尚未接入 agent。

## 验证结果

已在 mini-agentx Conda 环境中完成可编辑安装。

- --help 正常显示参数说明。
- --version 输出 mini-agentx 0.1.0。

## 下一步

完成配置与运行目录规范，再建立固定的推荐实验数据环境。