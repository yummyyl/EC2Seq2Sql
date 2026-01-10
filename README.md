# EC2Seq2Sql

This repository provides code to reproduce the **baseline experiments for Stage-1** (natural language eligibility text → lightweight structured snippet) and a **Stage-2 SQL generation agent** (lightweight structured snippet → executable SQL) used in our manuscript:

**EC2Seq2Sql: Patient-Trial Matching with LLM Agents**

> **Scope of this repo**
>
> - **Stage-1 (Baselines)**: Train and evaluate multiple baseline models (GPT-2, T5, BART, BioBERT, ClinicalBERT, TAPAS, GPT-3.5-turbo) to generate lightweight structured snippets from eligibility text.
> - **Stage-2 (SQL generation demo + interface)**: Generate schema-grounded SQL from Stage-1 snippets using a hierarchical prompt (system + human) and an LLM backend (LangChain OpenAI).  

---

## Installation

### 1) Create a Python environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## Data Preparation

We follow a non-redistribution strategy for upstream datasets:
We do not commit upstream dataset files.
We provide scripts to download the dataset from a pinned commit and generate deterministic splits.
See `data/README.md` for details.

### 1) Download the upstream seq2seq dataset

```bash
python scripts/download_seq2seq_data.py
```

Default output:  `data/raw/seq2seq_train.json`

### 2) Prepare deterministic train/valid/test splits

```bash
python scripts/prepare_splits.py
```

Default outputs:
`data/splits/train.jsonl`
`data/splits/valid.jsonl`
`data/splits/test.jsonl`
`data/splits/split_manifest.json`

## Stage-1: Train/Evaluate Baselines (Eligibility Text → Snippet)

Each baseline is implemented in:

```bash
baselines/<model>/train.py
```

All baselines follow a consistent interface and write results to:

```bash
outputs/<model>/metrics.json
outputs/<model>/predictions.jsonl 
```

Summarize Stage-1 results：

Once you have outputs/<model>/metrics.json files, run:

```bash
python scripts/run_stage1_baselines_summary.py
```

Outputs:

`outputs/stage1/table_stage1.csv`
`outputs/stage1/table_stage1.md`
`outputs/stage1/table_stage1.tex`

## Stage-2: Snippet → Schema-Grounded SQL

Stage-2 consumes Stage-1 output snippets (lightweight structured representations) and generates SQL queries that can be executed on an EHR database interface.

**Schema grounding (single-table interface)**

This repo provides a public demo schema (no patient rows, only column names): `data/schemas/demo_schema.json`

It matches a single-table database interface with columns:
`id, gender, condition, procedure, observation, laboratory, drug, birthday`

**Run Stage-2 SQL generation**

Prepare an input JSONL file where each line contains at least a snippet field:

Example data/stage2_inputs.jsonl:

```bash
{"id":"ex1","snippet":"age().num_filter(eq().op(GTEQ).val('18').temporal_unit(YEAR))"}
{"id":"ex2","snippet":"gender('female')"}
```

Run:

```bash
python scripts/run_stage2_sql_generation.py \
  --in_path data/stage2_inputs.jsonl \
  --schema_path data/schemas/demo_schema.json \
  --provider langchain_openai \
  --model gpt-4
```

Outputs:

`outputs/<model>/predictions.jsonl`
`outputs/<model>/metrics.json`

## Outputs

This repository keeps the `outputs/` directory in version control via `outputs/.gitkeep`.

By default, large model artifacts (checkpoints/weights) should not be committed.
We recommend committing only:

`outputs/**/metrics.json`

summary tables (`csv`, `md`, `tex`)

## Citation
If you found this work helpful, please consider citing us!

```bash
@article{EC2Seq2Sql,
  title={EC2Seq2Sql: Patient-Trial Matching with LLM Agents},
  author={Liu, Yang and Yongzhong, Han and Liang, Liu and Xiaoyan, Jiang and Ying, Li and Jihan, Huang and Qianmin, Su},
}
```

## License


## Contact

For questions or support, contact:

Qianmin Su: suqm@sues.edu.cn
