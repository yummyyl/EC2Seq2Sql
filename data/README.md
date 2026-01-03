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

Local directory layout
----------------------
data/
  raw/        # downloaded upstream data (NOT committed to git)
  splits/     # reproducible splits + manifest (can be committed)

Download the dataset
--------------------
From the repository root:

  python scripts/download_seq2seq_data.py

By default, the dataset is saved to:
  data/raw/seq2seq_train.json

Create reproducible splits (80/10/10)
-------------------------------------
From the repository root:

  python scripts/prepare_splits.py

Default settings:
- train_ratio = 0.8
- val_ratio   = 0.1
- test_ratio  = 0.1
- seed        = 42

Outputs (generated files)
-------------------------
The split script writes the following files to data/splits/:

  data/splits/train.jsonl
  data/splits/valid.jsonl
  data/splits/test.jsonl
  data/splits/split_manifest.json

Each JSONL line is one example with at least:
  - intent: natural language eligibility criterion
  - snippet: target DSL snippet
  - id: deterministic SHA256 hash derived from (intent + snippet)

Reproducibility notes
---------------------
1) The upstream dataset is pinned to a fixed commit, ensuring the same input file can be retrieved.
2) The split process is deterministic given the seed and ratios. The script also records:
   - SHA256 of the downloaded input file
   - SHA256 of each generated split file
   in data/splits/split_manifest.json

If you need to use a different location or seed, you can override:
  python scripts/download_seq2seq_data.py --out <PATH>
  python scripts/prepare_splits.py --in_path <PATH> --out_dir <DIR> --seed <INT>

Important note about "trial-aware" split
----------------------------------------
The upstream seq2seq file used here contains only (intent, snippet) pairs and does not include
a trial identifier. Therefore, the provided split is a deterministic example-level split.
(If a trial identifier mapping is available, the split script can be extended to support
trial-aware splitting.)
