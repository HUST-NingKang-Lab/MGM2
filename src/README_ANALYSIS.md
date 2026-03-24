# src / 当前实现分析

`src` 当前是一套以 masked abundance 建模为核心的预训练实现，默认做双任务 token 级监督，并可选增加样本级 `Qwen3` 对齐。

## 1. 模块职责

- `configs/Config.py`：严格 dataclass 配置，未知顶层键会 fail fast。
- `data/build_dataset.py`：把 CSV 或 BIOM OTU 表转换成轻量 pickle 数据集。
- `data/Preprocessing.py`：相对丰度的 log 预处理、样本内 quantile 分箱/固定阈值分箱，以及回归监督位选择。
- `data/Dataset.py`：动态 masking、padding、batch 组装，以及按 `sample_id` 关联 `Qwen3` 目标。
- `models/`：Transformer backbone、bin 头、回归头、可选 Qwen3 projection head。
- `losses/MultiTaskLoss.py`：bin loss、regression loss 和可选 alignment loss 组合。
- `training/TrainerModule.py`：Lightning 训练/验证/测试和保存加载。
- `training/Train.py`：CLI 入口、路径校验、dataloader 构建和 Trainer 创建。

## 2. 当前训练链路

```text
raw relative abundance
    -> build_dataset.py 保存原始相对丰度
    -> AbundancePreprocessor.transform()
    -> BinManager.transform()
    -> MicrobiomeDatasetMGM2
    -> abundance-only masking
    -> MGM2Model
    -> bin loss + regression loss
    -> optional CLS-to-Qwen3 contrastive loss
```

关键语义：

- 数据构建阶段保存的是原始相对丰度。
- 回归目标是预处理后的 log abundance。
- 被 mask 的位置保留 OTU identity，只腐蚀 abundance 输入。
- `bin_labels` 非监督位为 `-100`。
- `regression_mask` 只在回归监督位为 `True`。

## 3. Qwen3 对齐任务设计

- 输入文件来自 `qwen3_keywords_embedding.py`，HDF5 中至少包含 `embeddings` 与 `sample_ids`。
- 数据集按 `sample_id` 做 join，命中时返回：
  - `qwen3_embedding`
  - `qwen3_alignment_mask=True`
- 未命中的样本仍保留在 batch 中，只是 `qwen3_alignment_mask=False`。
- 模型始终输出最后一层 `cls_hidden`，只有开关开启时才增加 `qwen3_proj`。
- alignment loss 使用 batch 内 symmetric InfoNCE；有效配对数少于 2 时该项 loss 记 0。

## 4. 最近收紧的约束

- 旧 `presence` 任务与兼容字段已完全删除。
- `Train.py` 统一使用 `--ntv3_embedding_path` 与 `--qwen3_embedding_path` 两个显式路径参数。
- 开启 `Qwen3` 对齐却未提供文件、文件不存在、维度与 `projection_dim` 不匹配，都会尽早报错。
- `build_dataset.py` 在输出 0 个有效样本时会直接失败，避免后续统计阶段除零。

## 5. 仍需注意的点

- `Qwen3` 对齐依赖 batch 内有效样本数，如果有效样本稀疏，contrastive loss 会变稀。
- 当前 abundance 预处理仍采用 `log10(x) + 2` 再裁剪到 `[output_min, output_max]` 的历史方案；如果未来彻底改到纯 log 空间，需要同步修改回归目标和分箱阈值语义。
- `build_dataset.py` 的 `--ntv3_embedding` 参数名为兼容现有使用习惯而保留，训练入口则统一成 `*_path` 风格。

## 6. 总结

现在的 `src` 可以概括成：

**一个保留 OTU identity、只 mask abundance 输入、默认训练 masked bin classification 与 masked log-abundance regression，并可选利用 CLS 表征对齐 Qwen3 样本 embedding 的 Transformer 预训练实现。**
