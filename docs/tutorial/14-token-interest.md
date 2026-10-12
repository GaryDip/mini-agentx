# 14：yes/no单token置信评分与并发请求

## 这一章的开发流程与经验

先实测token和概率接口，再固定分数含义；并发应有请求预算、重试与缓存，不是把所有请求同时发送。

1. 要求恰好输出小写yes/no，不生成理由或JSON。
2. 一次调用读取同一位置的top_logprobs，检查两个选项都有有效概率。
3. 根据实际回答给置信度加正负号，再明确映射点击概率的规则。
4. 使用有界工作队列、单键锁、即时缓存与独立环境版本。
5. 验证真实并发与输出分布，不能把token置信度当作真实点击概率。

## 1. 实際评分定义

system完整内容在simulation/token_interest.py的SYSTEM_PROMPT，要求只选择yes或no、仅使用画像和电影元数据，不知道组别或分数。user一次只包含一个用户画像和一部电影。

DeepSeekClient.binary_logprobs发送thinking=disabled、temperature=1、max_tokens=4、logprobs=true、top_logprobs=20。实际输出必须是一个token；max_tokens=4给结束留余量，不意味着允许输出4个答案token。

真实探测发现temperature=0会使其他候选返回-9999哨兵，所以这里使用temperature=1。不能把哨兵当成真实概率或0。yes/no包含返回的单token大小写和空格变体，使用log-sum-exp等价的稳定归一化计算两类概率。

```text
q_yes = yes类概率 / (yes类概率 + no类概率)
回答yes：confidence=q_yes，signed_score=+confidence
回答no：confidence=1-q_yes，signed_score=-confidence
模拟兴趣概率=(signed_score+1)/2
```

例如yes置信0.9得到+0.9，no置信0.8得到-0.8。这按用户要求让实际回答决定正负；不是此前讨论的2q_yes−1分数。temperature=1的实际回答具有采样随机性，缓存固定第一次有效评价，避免重复请求筛选喜欢的结果。

点击兴趣概率的线性映射是模拟假设，并非真实点击校准。之后继续乘位置查看概率，并由程序采样点击。负分不会直接作为概率使用。

## 2. 概率检查

parse_binary要求输出和首token一致，只有一个输出token，yes与no都在返回候选中且有有限有效logprob。候选重复、缺少某个选项、哨兵或解释文本都不能作为有效评价。无效结果最多重试一次，仍失败则停止派发新任务，不能悄悄填0或重新使用旧分数。

每条缓存保留原始answer、logprobs、answer_confidence、q_yes、signed_score和用于点击的interest_probability。恢复时重新计算分数，检查缓存没有改变。

## 3. 高并发如何实现

score_token_pairs使用ThreadPoolExecutor。默认20并发，可设置1..64；只维护并发数规模的任务队列，完成一个再补一个。相同用户电影组合先去重，再通过文件锁协调多个进程，同一个缓存键不会并发发送重复请求。

429、网络错误、无效概率允许一次有限重试，并加入随机退避；其他HTTP4xx不继续重试。总预算按实际请求次数加锁计数，失败尝试也占预算。结果由主线程保存进度，成功评价逐条写入原子缓存。某个任务失败停止新增任务，已发出的请求仍收集并保存，不丢弃成功结果。

interest-token-cache与原JSON分数缓存分开；键包括画像、电影、提示词、解码设置、算法版本与模型名。改评分不会混用旧缓存。文件锁在进程退出时释放，锁文件本身可以保留。

## 4. 如何运行

使用新配置，不修改正在运行的旧JSON环境：

```bash
mini-agentx iterate \
  --baseline-run <当前基线ID> \
  --rounds 1 \
  --ab-config configs/abtest-profiles-token.toml
```

```toml
[feedback]
backend = "llm_profiles"
scoring = "yes_no_logprobs"
concurrency = 20
max_model_calls = 10000
```

其他字段沿用固定500用户画像、位置概率和评估器。500用户Top-10首轮最多5000个独立组合，单组合一次请求；10000是包含重试的兴趣调用上限，独立于workflow三个agent预算。并发减少等待时间，不减少计费请求。重复运行自动命中已完成的token缓存。

兴趣审计目录为interest-token-<id>，status记录model_calls、缓存数量、失败数量和并发配置，activity列出每个用户电影概率与正负分。原始概率响应单独保存。旧configs/abtest-profiles.toml继续使用批量JSON评分，不会因本次代码升级改变现有运行。

## 5. 实测与限制

真实12路并发验证interest-token-fef9ade431ea：12对、12请求，完整通过。所有样例回答yes，置信度大多接近1；这不是评分更有区分力的证据。极端自信时某个选项仍可能不进入Top-20，任务会按上述规则拒绝评分。

tests/test_token_interest.py验证yes正分/no负分、候选缺失拒绝、并发上限、去重、缓存恢复不调用API及预算耗尽后保留成功结果。旧画像与A/B流程继续保留。

## 原始训练历史的单电影抽查

score_token_pairs新增input_mode="history"，user消息改成user_id、training_history和item，不传画像摘要。历史和画像模式使用不同prompt，因此缓存不会混合。

真实100个独立用户抽查interest-token-b98e46f78327：每人20部训练历史＋1部电影，20路并发、100请求；实际推荐50样本中yes43/no7，低类型匹配50对照中yes13/no37。对照是刻意采样，不是A/B因果比较。约80%分数绝对值超过0.95，仍不能声称概率校准。默认A/B输入没有因此改变。
