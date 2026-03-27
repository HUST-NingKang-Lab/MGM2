# MGM2 `src` 包说明

`src/` 是当前 MGM2 的主实现。它不再兼容旧的 `presence` 任务，默认只做 token 级双任务 masked abundance pretraining，并可选增加样本级 `Qwen3` 对齐。

## 当前任务定义

默认开启的 token 级任务：

- masked abundance-bin prediction
- masked log-abundance regression

可选的样本级任务：

- `CLS -> Qwen3` contrastive alignment

其中 `Qwen3` 对齐默认关闭，只有显式开启时才会加载额外 HDF5 目标并增加 projection head 与 contrastive loss。

## 包结构

```text
src/
├── configs/Config.py
├── data/
│   ├── build_dataset.py
│   ├── Dataset.py
│   ├── Preprocessing.py
│   └── utils.py
├── losses/MultiTaskLoss.py
├── models/
│   ├── BaseModel.py
│   ├── Heads.py
│   └── Model.py
├── training/
│   ├── TrainerModule.py
│   └── Pretrain.py
└── __init__.py
```

## 最常用入口

### 构建数据集

```bash
python src/data/build_dataset.py   --otu_table data/otu_table.csv   --format csv   --ntv3_embedding ntv3_embeddings/otus_99_embeddings.h5   --output data/pretrain_dataset.pkl
```

### 预训练

```bash
python src/training/Pretrain.py   --data_file data/pretrain_dataset.pkl   --ntv3_embedding_path ntv3_embeddings/otus_99_embeddings.h5   --config src/configs/mgm2_default.json
```

### 启用 Qwen3 对齐

```bash
python src/training/Pretrain.py   --data_file data/pretrain_dataset.pkl   --ntv3_embedding_path ntv3_embeddings/otus_99_embeddings.h5   --enable_qwen3_alignment   --qwen3_embedding_path qwen3_embeddings/microbeatlas_keywords_emb.h5
```

## 配置说明

### `abundance`

- `min_abundance`：正丰度最小下界
- `output_min` / `output_max`：log 变换后的裁剪范围
- `normalize`：是否在 log 变换后再做标准化
- `num_bins`：bin 数量
- `binning_method`：`quantile`（样本内独立分箱）或 `fixed_threshold`
- `regression_top_k_bins`：纳入回归监督的最高 bin 数量
- `fixed_thresholds`：固定分箱阈值
- `enable_regression`：是否启用回归头

### `loss`

- `lambda_bin`：bin loss 权重
- `lambda_regression`：回归 loss 权重
- `bin_loss_type`：`cross_entropy` 或 `focal`
- `regression_loss_type`：`mse` 或 `mae`

### `training`

- `mask_prob`：监督 mask 概率
- `mask_ratio`：`(mask, random, keep)` abundance 腐蚀比例
- `batch_size` / `num_workers`
- `train_ratio` / `val_ratio`：预训练的 train/val 切分比例，要求和为 1
- `learning_rate` / `weight_decay`
- `warmup_steps` / `max_epochs`
- `devices` / `accelerator` / `precision`

### `qwen3_alignment`

- `enabled`：是否开启样本级对齐
- `embedding_path`：`qwen3_keywords_embedding.py` 产出的 HDF5 文件路径
- `projection_dim`：`CLS` 投影维度，必须和目标 embedding 维度一致
- `temperature`：InfoNCE 温度
- `loss_weight`：alignment loss 权重

目标 HDF5 需要包含：

- `embeddings`: `[num_samples, embedding_dim]`
- `sample_ids`: 与 `embeddings` 行对齐的样本 ID

## 训练语义

- 数据构建阶段只保存原始相对丰度。
- 训练阶段在 `Dataset.py` 中做 abundance log 预处理与分箱。
- masking 保留 OTU identity，只修改 abundance 通道。
- `bin` 与 `regression` 只在监督 mask 位置计算。
- `Qwen3` 对齐只在 batch 中存在有效 `sample_id` 匹配时计算；有效样本少于 2 时该项 loss 为 0。

## 当前的 fail-fast 约束

- 开启 `Qwen3` 对齐但未提供 `--qwen3_embedding_path` 会直接失败。
- `Qwen3` embedding 文件不存在会直接失败。
- `Qwen3` embedding 维度与 `projection_dim` 不一致会直接失败。
- 数据构建后如果没有任何有效样本，会直接失败而不是到统计阶段再除零。
