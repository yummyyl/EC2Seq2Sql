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

### 2) Install dependencies

```bash
pip install -r requirements.txt

### Data Preparation (Stage-1)


We follow a non-redistribution strategy for upstream datasets:

We do not commit upstream dataset files.

We provide scripts to download the dataset from a pinned commit and generate deterministic splits.

See data/README.md for details.

### 1) Download the upstream seq2seq dataset

```bash
python scripts/download_seq2seq_data.py

Default output: data/raw/seq2seq_train.json

### 2) Prepare deterministic train/valid/test splits

```bash
python scripts/prepare_splits.py

Default outputs:

data/splits/train.jsonl

data/splits/valid.jsonl

data/splits/test.jsonl

data/splits/split_manifest.json




