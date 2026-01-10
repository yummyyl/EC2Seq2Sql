Data (EC2Seq2Sql)
=================

This repository does NOT redistribute third-party dataset files.
Instead, we provide scripts to download the exact upstream version (pinned commit)
and to reproduce the train/valid/test splits used for baseline experiments.

Upstream dataset (pinned)
-------------------------
Source repository: uw-bionlp/clinical-trials-gov-data
Source file: data/seq2seq/train.json
Pinned commit: 62d54af08ba52e8196e664fcec01122a4d4e38ab

Permalink:
https://github.com/uw-bionlp/clinical-trials-gov-data/blob/62d54af08ba52e8196e664fcec01122a4d4e38ab/data/seq2seq/train.json

data/

README.md

raw/ # downloaded upstream files (NOT committed)

splits/ # generated splits (NOT committed by default)

schemas/ # schema summaries for Stage-2 SQL grounding (committed)


### `data/raw/` (downloaded; not committed)
- Stores the upstream dataset file downloaded by `scripts/download_seq2seq_data.py`.
- This directory is kept in the repo via `.gitkeep`, while its contents are ignored by `.gitignore`.

Expected file:
- `data/raw/seq2seq_train.json`

### `data/splits/` (generated; not committed by default)
- Stores the split files produced by `scripts/prepare_splits.py`.
- Files are written in **JSONL** format for streaming-friendly processing and reproducibility.

Expected files:
- `data/splits/train.jsonl`
- `data/splits/valid.jsonl`
- `data/splits/test.jsonl`
- `data/splits/split_manifest.json`

`split_manifest.json` records:
- upstream source (repo/path/commit)
- input SHA256 checksum
- split seed and ratios
- output file SHA256 checksums

> If you want maximum convenience (no re-splitting needed), you may commit the split files.
> By default, we do not commit them.

### `data/schemas/` (committed)
- Contains **schema-only** JSON files used to ground Stage-2 (snippet → SQL) generation.
- These files contain **only table/column names** and **no patient-level records**.

Provided file:
- `data/schemas/demo_schema.json`

---

## Upstream Dataset (Stage-1: Eligibility text → Snippet)

Stage-1 baselines use a seq2seq-style dataset where each example contains:

- `intent`: natural language eligibility criterion text
- `snippet`: lightweight structured snippet (target)

Example:
```json
{
  "intent": "- 18 years or older",
  "snippet": "age().num_filter(eq().op(GTEQ).val('18').temporal_unit(YEAR))"
}
```

## How to Prepare the Data

### 1) Download the upstream dataset

`python scripts/download_seq2seq_data.py`

Default output:

`data/raw/seq2seq_train.json`

The script prints the SHA256 checksum after download.

### 2) Create deterministic train/valid/test splits

`python scripts/prepare_splits.py`

Default outputs:

`data/splits/train.jsonl`

`data/splits/valid.jsonl`

`data/splits/test.jsonl`

`data/splits/split_manifest.json`

The split method is a deterministic shuffle with a fixed seed (default: 42).
All split parameters and file hashes are recorded in `split_manifest.json`.

## Stage-2 Schema (Snippet → SQL)

Stage-2 takes the Stage-1 output snippet and generates schema-grounded SQL queries.

In our manuscript setting, the target database interface is a single table named patients with the following columns:

`id` 、 `gender` 、 `condition` 、 `procedure` 、 `observation` 、 `laboratory` 、 `drug` 、 `birthday`

Accordingly,` data/schemas/demo_schema.json` provides a public demo schema that contains:

- table name and column names only

- no data rows

- no private or hospital EHR content

You may replace this schema file with your own schema JSON if your database interface differs.

## Notes on Redistribution
- This repo does not include the upstream dataset file.

- Users should run the scripts to download and prepare data locally.

- If you are the data owner or have explicit permission to redistribute, you may commit split files for convenience.

## Troubleshooting
- If download fails, check network access and GitHub raw content availability.

- If splitting fails, ensure the input file is a JSON list and each example contains intent and snippet.

- For reproducibility, keep the upstream commit SHA and split seed unchanged (both are recorded in `split_manifest.json`).
