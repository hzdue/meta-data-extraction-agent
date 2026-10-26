# ME Agent

基于原文证据的科学论文元数据提取、验证与修复系统。

本目录是已实测的 **GPU3 最佳基线**，不包含尚未验证的 v20 开发改动。主入口为 `extract.py` 和 `run.sh`。

## 实测成绩

模型：`google/gemma-4-31B-it`。相同的 30 篇论文，复用已有 seed、anchor 和 detector，仅重新运行 repair。

| 阶段 | 设施 F1 | 样品 F1 | 数据集 F1 | PaN Mean |
|---|---:|---:|---:|---:|
| Global seed | 89.67% | 92.04% | 76.28% | 86.00% |
| GPU3 final | 90.22% | 92.96% | 79.91% | **87.70%** |

PaN Mean 是每篇论文三个实体 F1 的等权平均，再对全部论文取平均。

- [逐篇成绩](benchmarks/gpu3_30papers.json)
- [代码来源和校验记录](BASELINE.json)
- [固定论文清单](qwen30_papers.txt)

这是已有运行的测量结果，整理发布目录时没有重新调用模型。这批论文已用于开发，并非未见测试集。从头重新推理不保证得到完全相同的 seed 或最终分数。

## 系统流程

原文 → global seed → anchor evidence 与覆盖检查 → detector → repair → 最终预测。

当前 repair 包括有界证据检索、新增实体证据审核、关系候选排序、响应格式适配和事务校验。每次最多补充两个原文片段，每篇最多 12 个。共现只用于排序，不直接证明关系。

本版不含 v20 的数据集重新分组、新报告范围门控或 10 篇开发集改动。Gold 仅用于评估，不进入提取或 repair 的推理过程。

## 环境与数据

推理使用 Linux、NVIDIA GPU 和 Bash。已有成功运行的服务器使用 Python 3.12、vLLM 0.22.1 和单卡 A100 80GB；这不是其他硬件的最低要求声明。

`requirements.txt` 是依赖声明，不是完整环境锁文件。Transformers、CUDA 和模型文件的完整版本未记录，建议复用已经成功运行 Gemma 的服务器环境。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

准备与论文清单对应的数据：

```text
SOURCES/input_papers/<paper_id>/paper.pdf
SOURCES/gold_data/<paper_id>.json
```

本包不附带论文、gold、模型权重或运行缓存。批处理会执行评估，因此需要 gold 文件；提取和 repair 不会将 gold 提供给模型。

输入和 gold 目录已被 `.gitignore` 排除。评估器及对齐配置按基线保留。Schema 原始字节受 `.gitattributes` 保护，以保持已有缓存的指纹一致。

## 单 GPU 从头运行

在本目录执行：

```bash
GPU_ID=0 PYTHON=python3 MODEL_PROFILE=gemma31b \
MODEL=google/gemma-4-31B-it REPAIR_MODEL=google/gemma-4-31B-it \
SINGLE_GPU_PIPELINE=1 VLLM_PORT=18020 \
ME_OUTPUT_ROOT=outputs ME_RUN_NAME=gemma31b_baseline \
PAPER_LIST=qwen30_papers.txt BATCH_SPLIT=all MAX_PAPERS=30 \
BATCH_RUN_ID=run_baseline30_gemma31b \
TOOL_TRANSPORT=local ENABLE_TRACING=0 \
FRESH_RUN=1 SKIP_COMPLETED=0 REPAIR_FROM_CACHE=0 \
REPAIR_MAX_OUTPUT_TOKENS=6500 REPAIR_MAX_ROUNDS=2 \
REPAIR_MEASUREMENT_RECONCILIATION=1 \
bash run.sh
```

所有阶段均使用 Gemma。每篇论文的输出位于 `outputs/gemma31b_baseline/<paper_id>/`；批次监控记录在相应的 `error_analysis/<batch_id>/` 下。

## 复用缓存，只运行 repair

先把已有四个缓存文件放入目标目录：

```text
outputs/gemma31b_baseline/<paper_id>/checkpoint.json
outputs/gemma31b_baseline/<paper_id>/global_seed_prediction.json
outputs/gemma31b_baseline/<paper_id>/global_seed_anchors.json
outputs/gemma31b_baseline/<paper_id>/global_seed_anchor_detector.json
```

仍需相同的原文和 schema，程序会核对指纹。然后执行：

```bash
GPU_ID=0 PYTHON=python3 MODEL_PROFILE=gemma31b \
MODEL=google/gemma-4-31B-it REPAIR_MODEL=google/gemma-4-31B-it \
ME_OUTPUT_ROOT=outputs ME_RUN_NAME=gemma31b_baseline \
PAPER_LIST=qwen30_papers.txt BATCH_SPLIT=all MAX_PAPERS=30 \
BATCH_RUN_ID=run_baseline30_repair VLLM_PORT=18020 \
REPAIR_UPSTREAM_ROOT= REPAIR_IN_PLACE=1 \
ENABLE_TRACING=0 TOOL_TRANSPORT=local \
REPAIR_MAX_OUTPUT_TOKENS=6500 REPAIR_MAX_ROUNDS=2 \
REPAIR_MEASUREMENT_RECONCILIATION=1 \
bash run.sh
```

此命令保留 seed、anchor 和 detector，覆盖目标目录中的 repair 和最终结果。需要保留对照时，请先另存原结果。新实验应使用新的批次 ID，端口必须可用。

## 离线检查

```bash
python -m unittest discover -s tests -q
bash -n run.sh
python extract.py --help
```

测试不启动模型服务。发布整理时，235 项测试通过、3 项跳过；30 篇论文的 33 个修复事务重放结果与已有 GPU3 结果一致。

## 目录结构

```text
extract.py                    主程序与 repair
run.sh                        单 GPU 批处理入口
requirements.txt              运行依赖
qwen30_papers.txt              固定 30 篇清单
BASELINE.json                 来源、校验值和验证记录
SOURCES/schemas/              Schema 与 seed 提示模板
SOURCES/evaluation/           评估器及对齐配置
benchmarks/gpu3_30papers.json  汇总与逐篇成绩
tests/                        离线测试
```

本目录没有嵌套 `.git`，可作为新 GitHub 仓库的根目录。README 使用 UTF-8 编码；普通文本文件使用 LF 换行，schema 保留原始字节。
