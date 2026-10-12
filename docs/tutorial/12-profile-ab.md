# 12：画像与电影兴趣评价，接入模拟A/B

## 这一章的开发流程与经验

用户画像、兴趣判断、曝光和统计是四件不同的事。先固定画像与评分规则，再让程序产生曝光点击，最后由Evaluation分析汇总。

1. 验证固定500画像、数据版本和电影映射。
2. 在隔离进程中让两模型生成真实Top-10排名。
3. 固定用户分组，只评价对应模型实际曝光的用户电影对。
4. LLM只看画像和电影元数据，输出兴趣概率与公开理由。
5. 校验并缓存用户电影评价，程序处理位置和点击采样。
6. 沿用用户级统计、护栏及Evaluation，不预设一定KEEP。

经验：LLM不是真实用户，兴趣概率未经校准。这里学习可解释、可追溯的反馈环境，不将它当作线上收益。画像环境独立于验证集，验证指标继续单独报告。

## 1. 两个LLM角色

simulation/interest.py 的兴趣评价器接收固定画像和候选电影，不知道组别、模型得分、排名及留出答案。Evaluation Agent接收A/B汇总，不读取私有逐电影概率。

system完整模板保存于interest-<id>/prompt.md：仅根据画像和电影标题、日期、类型，不补剧情、不查外部知识。输出interest_probability在0..1，reason简短说明匹配或不确定。

user输入格式：

```json
{"cases":[{"user_id":0,"profile":{"summary":"...","preferences":[],"uncertainties":[]},"items":[{"item_id":12,"title":"...","release_date":"...","genres":["Comedy"]}]}]}
```

结果：

```json
{"ratings":[{"user_id":0,"item_id":12,"interest_probability":0.7,"reason":"类型符合已观察的偏好"}]}
```

数字只是格式示例。每批最多25个用户电影对；校验ID完整且不重复，概率为有限数且不能是bool，失败最多修复一次。

## 2. 画像与缓存怎样固定

load_population确认500画像生成已完成，users/items和核心prompt摘要一致，每人输入身份和偏好引用有效。首次接入保存population_snapshot.json，后续修改画像会拒绝，不静默混用环境。

interest-cache/<SHA256>.json 的键由画像内容、电影元数据、评价prompt和模型名决定，不含组别或排名。相同用户电影组合跨候选复用。新内容生成新的键。当前模型名是配置标识，服务端同名模型升级仍可能改变未缓存评价；实际调用返回保存在请求审计目录，严格复现实验应固定同一批缓存。

缓存不是新的独立用户证据；重复曝光也不会增加独立用户数量。每次interest-<id>保存请求、返回、activity.md和预算状态；预算耗尽或批次失败时已验证的电影评价保留，下次A/B复用这些缓存。

## 3. 点击如何产生

固定画像用户用内部user_id和实验seed分组。每人仅使用自己的A或B排名。Top-10排除训练已交互电影。默认20次曝光，重复展示两遍列表。

第r个位置的查看概率设为 1 / r^position_exponent，默认指数0.5。兴趣与查看相乘，再由现有固定用户随机数采样点击。错误事件强制不点击。

这是人为定义的位置假设，不是学到的真实查看曲线。延迟和错误率仍是旧模拟器生成的值，不是模型推理耗时。置信区间仍按用户bootstrap。

## 4. 如何切换环境

原configs/abtest.toml继续使用随机合成偏好。新configs/abtest-profiles.toml启用：

```toml
[feedback]
backend = "llm_profiles"
profiles_directory = "runs/profiles-<时间戳>-<id>"
users = 500
llm_config = "configs/interest-llm.toml"
max_model_calls = 240
position_exponent = 0.5
```

画像目录必须指向已生成并完成的批次。users可以设小规模先验证，但200用户/组的现有门槛不放宽，小样本通常EXTEND。profiles配置中的simulation.users不决定人数，feedback.users决定从固定画像池抽取多少人。

```bash
mini-agentx iterate \
  --baseline-run baseline-c30ead6da742 \
  --rounds 1 \
  --ab-config configs/abtest-profiles.toml
```

500用户、每人Top-10意味着最多5000个兴趣组合，首次正常需要约200次批量请求；240是每个候选A/B的兴趣请求上限，修复另占次数。兴趣请求使用独立预算，不包含在workflow的三个agent调用预算中。后续候选有多少新增请求取决于推荐电影重叠与缓存。不能把workflow.model_calls当成全部LLM调用总数；各A/B的models.interest_calls记录兴趣请求数。

## 5. 报告与限制

A/B报告维持mode=models和原事件字段，因此现有Evaluation及反馈存储可直接接收。models记录画像来源、兴趣模型、prompt摘要、缓存数量、调用次数和位置假设；新运行另记录兴趣评价摘要及审计目录。私有兴趣理由与概率不注入Brainstorm的观测证据。

首次真实20用户验证 ab-e837c6f8f112：400次曝光，9次兴趣请求，B−A模拟CTR差值约+0.06515，95%区间约[-0.01871,+0.15711]，样本门槛不通过。只验证执行，不证明提升。

现有最小增益、分流和护栏要求不变。LLM对类型匹配和知名标题的偏好仍可能构成模拟偏差；概率缺少真实点击校准。此环境没有使用验证或测试答案，也不将负反馈从没有出现的类型中推断出来。
