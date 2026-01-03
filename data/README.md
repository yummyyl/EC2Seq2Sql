Data for EC2Seq2Sql
===================

This repository uses a public seq2seq dataset (intent -> DSL snippet pairs) released by:
uw-bionlp/clinical-trials-gov-data

Source (pinned version)
-----------------------
Repository: uw-bionlp/clinical-trials-gov-data
File: data/seq2seq/train.json
Pinned commit: 62d54af08ba52e8196e664fcec01122a4d4e38ab

Permalink:
https://github.com/uw-bionlp/clinical-trials-gov-data/blob/62d54af08ba52e8196e664fcec01122a4d4e38ab/data/seq2seq/train.json

Download
--------
Run the script:
  python scripts/download_seq2seq_data.py

The dataset will be saved to:
  data/processed/ec2dsl_pairs.json

Splits (for reproducibility)
----------------------------
We provide fixed data splits under:
  data/splits/

These splits are used to reproduce baseline results reported in the paper (e.g., Table 4).
