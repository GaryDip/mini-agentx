# 13：固定数据与推荐输出，让模型自己训练

## 这一章的开发流程与经验

先明确输入与推理输出，再决定哪些状态必须保存。不要让框架访问模型内部embedding名称；非梯度模型也应该能够工作。

1. 固定训练输入，禁止把验证/测试答案或模拟用户画像当特征。
2. 模型自行拥有表示、优化器、损失、采样和正则化。
3. 用统一打分接口接入验证与推荐。
4. 在小数据上检查训练、打分一致性、实际权重序列化与恢复。
5. 保留旧模型兼容桥，验证旧结果不变。

## 1. 输入是什么

新候选model.py定义Recommender(num_users,num_items,settings)，继承torch.nn.Module。settings是当前训练配置，包含资源上限与现有参数。fit_epoch收到训练正反馈(user_id,item_id)列表、配置副本和固定种子的random.Random。

用户历史可以由训练对构建，例如图邻接或用户交互列表。当前接口没有额外传电影内容、评分或时间戳。用户画像和点击概率属于反馈环境，不作为训练输入。数据划分、评估器、外部预算不允许模型修改。

## 2. 模型接口

```python
class Recommender(torch.nn.Module):
    def __init__(self, num_users, num_items, settings):
        ...  # 自己决定参数、buffer和网络结构

    def fit_epoch(self, train_pairs, settings, rng):
        ...  # 执行一轮自有训练，返回有限标量loss或诊断值

    def score(self, user_ids, item_ids):
        ...  # 两个同长度整数Tensor，输出同长度一维分数

    def score_all(self):
        ...  # 输出[num_users,num_items]有限分数矩阵
```

分数越高越推荐，score必须与score_all相应位置一致。模型不需要users/items属性，也不需要embedding。模型训练目标的诊断值不是跨模型可直接比较的质量指标；模型比较仍使用统一验证指标与A/B。

nn.Module用于统一state_dict保存，训练参数或推理buffer都可以。框架用torch.save和weights_only=True加载；所有推理依赖状态必须进入state_dict，不能只保留在普通Python属性或优化器中。构造新实例后加载权重即可打分；当前不支持优化器断点续训。

## 3. 框架还负责什么

recommender/train.py保留固定种子、线程、外部epoch循环、验证评估、按NDCG选最佳权重及报告。它调用model.fit_epoch，而不读取embedding或计算候选损失。新的training_strategy报告标为model_owned，避免把某个配置字符串当成已实现算法。

model.py可自定义损失与正则，也可使用recommender/strategy.py的现有函数。training.toml仍保持现有数值资源字段及seed/loss/negative_sampling；暂未增加任意模型配置区。模型应说明如何解释参数，不支持的组合应报错，不能假装配置生效。

同源码配置种子去重和已验证历史产物复用继续工作。框架的资源限制不因模型拥有训练而取消。

## 4. 两个示例

- examples/models/popularity.py：没有embedding和optimizer，fit_epoch统计训练物品频次，作为buffer保存，输出热门排序。
- examples/models/mlp.py：内部user_latent/item_latent加MLP，拥有自己的优化器和BPR训练，仅实现bpr/uniform配置。score_all分块计算，避免一次建立全部用户物品对的中间激活。

它们是接口与实现示例，不声称优于基线。Brainstorm可通过目录和搜索工具阅读，Developing可以在独立候选model.py实现其他结构。Item-KNN、图模型等可在这个接口下实现；序列模型可能还需要后续明确时序输入协议。

## 5. 检查与A/B加载

developing_worker调用interface.check_contract：构造3用户5物品，执行一次fit_epoch，再检查分数形状、有限值和一致性；实际写入内存中的权重文件，用weights_only=True恢复，新实例输出须一致。非梯度模型不强制backward。

隔离ranking_worker通过model_class选择新Recommender或旧MatrixFactorization，再用完整training_settings构造，加载保存的state_dict。score_all先clone再屏蔽训练已见物品，避免热门模型expand共享buffer被掩码修改。Top-10排序、去重和已见检查仍由框架执行。

独立evaluate命令的任意候选加载仍有限制；自定义模型通过开发内验证与隔离A/B评估，不在主进程直接import不受信候选。

## 6. 旧模型如何兼容

原recommender/model.py未改，避免破坏旧报告的源码摘要。旧MatrixFactorization(users,items,dimensions)由interface.LegacyTraining执行原训练逻辑；users/items依赖只存在兼容桥。旧权重与候选仍可加载。

真实回归验证：原始基线20epoch NDCG仍为0.04143242698221069。无embedding的Popularity在隔离环境完成训练，NDCG为0.0319165438343444，并成功恢复生成942用户Top-10。它验证接口通用性，不是质量提升。

## 7. 下一次运行

Brainstorm任务边界和Developing提示词已经更新，能提出新Recommender架构。无需更改iterate命令；如要使用画像反馈继续传--ab-config configs/abtest-profiles.toml。是否选新架构仍由提案选择决定，不保证每轮都会修改模型。
