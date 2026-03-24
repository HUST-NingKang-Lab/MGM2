# MGM2 仓库说明

`MGM2` 是一个面向微生物组 OTU 表的预训练项目。当前主实现位于 [src](/data6/zhanghaohong/projects/MWorld/MGM2/src)，核心是：

- 基于 `NTv3` OTU embedding 的 Transformer backbone
- masked `bin` 预测
- masked `log-abundance` 回归（默认监督最高 1 个 bin，可配置扩大到 top-k bins）
- 可选的 `CLS -> Qwen3` 样本级对比对齐任务

旧版实验代码保留在 `legacy_code/`，当前开发和使用都应以 `src/` 为准。

## 目录

```text
MGM2/
├── src/                     # 当前主实现
│   ├── configs/            # 严格配置定义
│   ├── data/               # 数据构建、预处理、dataset、loader
│   ├── losses/             # 多任务损失
│   ├── models/             # Backbone 与 heads
│   ├── training/           # Lightning 训练入口
│   ├── README.md
│   └── README_ANALYSIS.md
├── tests/                  # 当前回归测试
├── docs/                   # 设计文档、计划、仓库说明
├── legacy_code/            # 历史代码，仅存档
├── ntv3_embeddings/        # NTv3 embedding 文件
├── qwen3_embeddings/       # Qwen3 样本 embedding 文件
└── data/                   # 预处理数据与原始 OTU 表
```

## 当前训练任务

默认训练是双任务 masked pretraining：

1. `bin` 分类：预测被监督 mask 位置的丰度区间。
2. `regression` 回归：预测被监督 mask 位置的 log 变换后丰度。

可选第三项不是 token 级任务，而是样本级对齐任务：

3. `qwen3_alignment`：用最后一层 `CLS` 表征和外部 `Qwen3` 关键词 embedding 做 batch 内对比学习。

## 快速开始

### 1. 构建预训练数据集

```bash
python src/data/build_dataset.py   --otu_table data/otu_table.csv   --format csv   --ntv3_embedding ntv3_embeddings/otus_99_embeddings.h5   --output data/pretrain_dataset.pkl
```

如果输入是 BIOM：

```bash
python src/data/build_dataset.py   --otu_table data/otu_table.biom   --format biom   --ntv3_embedding ntv3_embeddings/otus_99_embeddings.h5   --output data/pretrain_dataset.pkl
```

### 2. 启动训练

```bash
python src/training/Train.py   --data_file data/pretrain_dataset.pkl   --ntv3_embedding_path ntv3_embeddings/otus_99_embeddings.h5   --config src/configs/mgm2_default.json
```

启用 `Qwen3` 对齐：

```bash
python src/training/Train.py   --data_file data/pretrain_dataset.pkl   --ntv3_embedding_path ntv3_embeddings/otus_99_embeddings.h5   --enable_qwen3_alignment   --qwen3_embedding_path qwen3_embeddings/microbeatlas_keywords_emb.h5
```

## 关键约束

- `src/` 已经彻底移除旧 `presence` 任务。
- masking 只腐蚀 abundance 通道，保留 OTU identity embedding。
- 回归目标是预处理后的 `log-abundance`，不是原始相对丰度。
- `Qwen3` 对齐开启时，样本缺少对应 `sample_id` 不会被丢弃，只是不计算该条 alignment loss。
- 如果 `Qwen3` embedding 文件不存在或维度与配置不匹配，当前实现会尽早报错。

## 文档入口

- [src/README.md](/data6/zhanghaohong/projects/MWorld/MGM2/src/README.md)：当前 `src` 包的使用说明。
- [src/README_ANALYSIS.md](/data6/zhanghaohong/projects/MWorld/MGM2/src/README_ANALYSIS.md)：当前实现分析与设计说明。
- [docs/repository_guide_zh.md](/data6/zhanghaohong/projects/MWorld/MGM2/docs/repository_guide_zh.md)：详细中文仓库导览。
