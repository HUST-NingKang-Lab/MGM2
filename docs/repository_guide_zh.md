# MGM2 仓库中文导览

## 1. 仓库定位

这个仓库用于训练微生物组基础模型。当前真正应该使用的是根目录下的 [src](/data6/zhanghaohong/projects/MWorld/MGM2/src) 包，而不是历史目录 `legacy_code/`。

当前主线目标不是传统语言模型里的 token identity 重建，而是围绕 OTU abundance 做 masked pretraining：

- 对被选中的 OTU 位置预测 abundance bin
- 对被选中的 OTU 位置回归 log 变换后的 abundance
- 可选地把整条样本的 `CLS` 表征对齐到外部 `Qwen3` 样本 embedding

## 2. 仓库结构

```text
MGM2/
├── src/
│   ├── configs/
│   ├── data/
│   ├── losses/
│   ├── models/
│   ├── training/
│   ├── README.md
│   └── README_ANALYSIS.md
├── tests/
├── docs/
├── data/
├── ntv3_embeddings/
├── qwen3_embeddings/
└── legacy_code/
```

各目录职责：

- `src/configs/`：严格配置对象定义。
- `src/data/`：数据构建、预处理、dataset、collator、dataloader。
- `src/models/`：backbone 和任务头。
- `src/losses/`：多任务损失。
- `src/training/`：Lightning 模块与训练 CLI。
- `tests/`：当前回归测试，重点是 `test_mgm2_regressions.py`。
- `docs/`：设计文档、实施计划和仓库说明。
- `legacy_code/`：历史版本，仅供参考，不建议继续扩展。

## 3. 当前模型做什么

### 3.1 默认任务

默认有两个 token 级任务：

1. `bin` 预测
把预处理后的 abundance 分到若干离散 bin 中，在被监督的 mask 位置做分类。

2. `regression` 回归
对同一批监督位回归预处理后的连续 abundance 值。

这里的回归目标不是原始相对丰度，而是经过 `AbundancePreprocessor` 处理后的值，目前实现仍是 `log10(x) + 2` 后裁剪到配置范围。

### 3.2 可选任务

3. `Qwen3` 对齐
如果开启 `qwen3_alignment`，模型会取最后一层 `CLS` hidden state，经 projection head 投影到目标空间，再和外部 `Qwen3` embedding 做对比学习。

## 4. 数据流

### 4.1 原始输入

模型直接依赖两类输入：

- OTU 表
- NTv3 embedding 文件

可选还会依赖第三类输入：

- Qwen3 样本 embedding 文件

### 4.2 数据集构建

执行脚本： [build_dataset.py](/data6/zhanghaohong/projects/MWorld/MGM2/src/data/build_dataset.py)

作用：

- 读取 CSV 或 BIOM OTU 表
- 计算相对丰度
- 每个样本只保留非零 OTU
- 把 OTU ID 映射到 NTv3 索引
- 保存轻量 pickle 数据集

输出数据集里保存的是：

- `samples`
- `otu_data`
- `otu_id_to_idx`
- `ntv3_embedding_path`
- `format_version`

注意：这里保存的是原始相对丰度，而不是 log 后结果。真正的 abundance 预处理发生在训练数据集阶段。

### 4.3 训练时的数据处理

核心文件： [Dataset.py](/data6/zhanghaohong/projects/MWorld/MGM2/src/data/Dataset.py)

训练时会做这些事：

- 读取 pickle 数据集
- 动态加载 NTv3 embedding
- 对 abundance 做 log 预处理与分箱
- 构造 `bin_labels` 与 `regression_targets`
- 进行动态 masking
- 可选按 `sample_id` 关联 `Qwen3` embedding
- 经 collator padding 后组 batch

当前 masking 设计的关键点：

- 保留 OTU identity
- 只腐蚀 abundance 通道
- 不再清零 embedding

这点很重要，因为当前任务不是“猜这个位置是哪一个 OTU”，而是“已知这个位置是哪个 OTU，预测它的 abundance”。

## 5. 预处理与分箱

核心文件： [Preprocessing.py](/data6/zhanghaohong/projects/MWorld/MGM2/src/data/Preprocessing.py)

### 5.1 abundance 预处理

当前流程大致是：

1. 负值和非法值归零
2. 正值但小于 `min_abundance` 的部分抬到下界
3. 对正值做 `log10(x) + 2`
4. 裁剪到 `[output_min, output_max]`
5. 如果开启 `normalize`，再做标准化

### 5.2 分箱

`BinManager` 支持两种方式：

- `quantile`
- `fixed_threshold`

默认一般使用 `quantile`，并且是按样本内分位数独立分箱。分箱结果用于 token 级分类任务，而回归监督位由 `regression_top_k_bins` 控制，默认只回归最高 1 个 bin。

## 6. 模型结构

核心文件：

- [BaseModel.py](/data6/zhanghaohong/projects/MWorld/MGM2/src/models/BaseModel.py)
- [Model.py](/data6/zhanghaohong/projects/MWorld/MGM2/src/models/Model.py)
- [Heads.py](/data6/zhanghaohong/projects/MWorld/MGM2/src/models/Heads.py)

结构可以理解为：

```text
OTU embedding + abundance value
    -> input projection
    -> Transformer backbone
    -> token heads / optional CLS head
```

具体输出：

- `bin_logits`
- `regression_preds`（如果启用回归）
- `cls_hidden`
- `qwen3_proj`（如果启用 Qwen3 对齐）

## 7. 损失函数

核心文件： [MultiTaskLoss.py](/data6/zhanghaohong/projects/MWorld/MGM2/src/losses/MultiTaskLoss.py)

总损失由三部分组成：

- `bin_loss`
- `regression_loss`
- `qwen3_alignment_loss`（可选）

默认不开启 `Qwen3` 对齐时，总损失就是双任务损失。

`Qwen3` 对齐目前使用 batch 内 symmetric InfoNCE：

- 正样本：同一样本的 `CLS` 投影和 `Qwen3` embedding
- 负样本：同一 batch 内其他有效样本
- 若 batch 内有效对齐样本少于 2 条，则该 loss 记为 0

## 8. 预训练入口

核心脚本： [Pretrain.py](/data6/zhanghaohong/projects/MWorld/MGM2/src/training/Pretrain.py)

基础训练示例：

```bash
python src/training/Pretrain.py   --data_file data/pretrain_dataset.pkl   --ntv3_embedding_path ntv3_embeddings/otus_99_embeddings.h5   --config src/configs/mgm2_default.json
```

启用 `Qwen3` 对齐：

```bash
python src/training/Pretrain.py   --data_file data/pretrain_dataset.pkl   --ntv3_embedding_path ntv3_embeddings/otus_99_embeddings.h5   --enable_qwen3_alignment   --qwen3_embedding_path qwen3_embeddings/microbeatlas_keywords_emb.h5
```

预训练流程只使用 train/val 切分，不跑测试集。

`Pretrain.py` 现在已经补了几类 fail-fast：

- `data_file` 不存在直接失败
- `ntv3_embedding_path` 不存在直接失败
- 开启 `Qwen3` 对齐但未提供文件直接失败
- `qwen3_embedding_path` 不存在直接失败

## 9. Qwen3 文件格式

当前实现假定 `qwen3_keywords_embedding.py` 产出的 HDF5 至少包含：

- `embeddings`
- `sample_ids`

其中：

- `embeddings.shape = [num_samples, embedding_dim]`
- `sample_ids[i]` 对应 `embeddings[i]`

数据集在训练时会按 `sample_id` 做 join。找不到匹配目标的样本不会被删除，只是不参与 alignment loss。

## 10. 配置系统

核心文件： [Config.py](/data6/zhanghaohong/projects/MWorld/MGM2/src/configs/Config.py)

主要配置块：

- `abundance`
- `model`
- `loss`
- `training`
- `qwen3_alignment`

特点：

- 使用 dataclass
- 顶层未知字段直接报错
- 已经不再吞旧 `presence` 配置

这意味着如果你把旧版三任务配置直接喂给当前实现，会马上失败，而不是悄悄忽略。

## 11. 当前已经修过的一些关键问题

这轮之前，`src` 这条线已经清理过一批实质性问题，包括：

- 删除旧 `presence` 任务
- 修正 masking 设计，保留 OTU identity
- 修正 CSV 路径下相对丰度保存错误
- 修正 collator 对 regression mask 的 padding 问题
- 修正 `Train.py` 包路径与 CLI 标志漂移
- 增加可选 `Qwen3` 对齐任务
- 增加 `Qwen3` 路径与维度的 fail-fast 检查
- 修正 `build_dataset.py` 在 0 有效样本时的晚期报错

## 12. 当前使用建议

如果你准备继续在这个仓库上开发，建议遵循这些原则：

- 所有新功能都基于 `src/`，不要继续往 `legacy_code/` 加内容。
- 新增任务前，先确认它是 token 级还是 sample 级，避免再混回旧版三任务思路。
- 如果引入新的样本级对齐目标，优先沿用现有 `sample_id -> external embedding` 这条接入方式。
- 改动训练入口时，优先保持 CLI 路径语义一致，比如统一使用 `*_path`。
- 改动数据构建逻辑时，优先保证“构建阶段只存原始相对丰度，训练阶段再做预处理”的分层不被打乱。

## 13. 一句话总结

这个仓库现在最准确的理解方式是：

**一个基于 OTU + NTv3 embedding 的微生物组 Transformer 预训练项目，默认做 masked bin 分类和 masked log-abundance 回归，并可选用 CLS 表征对齐外部 Qwen3 样本 embedding。**
