import argparse
import hashlib
import json
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Tuple


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_example(ex: Dict[str, Any]) -> Dict[str, Any]:
    
    if "intent" not in ex or "snippet" not in ex:
        raise ValueError("Each example must contain 'intent' and 'snippet'.")
    return ex


def example_id(ex: Dict[str, Any]) -> str:
    
    base = f"{ex.get('intent','')}\n{ex.get('snippet','')}"
    return sha256_text(base)


def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON list of examples.")
    return [normalize_example(x) for x in data]


def split_indices(n: int, train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str, default="data/raw/seq2seq_train.json")
    ap.add_argument("--out_dir", type=str, default="data/splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--commit_sha", type=str, default="62d54af08ba52e8196e664fcec01122a4d4e38ab")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = load_json_list(args.in_path)

   
    for ex in data:
        ex.setdefault("id", example_id(ex))

    train_idx, val_idx, test_idx = split_indices(len(data), args.train_ratio, args.val_ratio, args.seed)

    train = [data[i] for i in train_idx]
    valid = [data[i] for i in val_idx]
    test = [data[i] for i in test_idx]

    train_path = os.path.join(args.out_dir, "train.jsonl")
    valid_path = os.path.join(args.out_dir, "valid.jsonl")
    test_path = os.path.join(args.out_dir, "test.jsonl")
    manifest_path = os.path.join(args.out_dir, "split_manifest.json")

    write_jsonl(train_path, train)
    write_jsonl(valid_path, valid)
    write_jsonl(test_path, test)

    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source": {
            "repo": "uw-bionlp/clinical-trials-gov-data",
            "path": "data/seq2seq/train.json",
            "commit_sha": args.commit_sha,
            "input_file": args.in_path,
            "input_sha256": sha256_file(args.in_path),
        },
        "split": {
            "method": "deterministic_shuffle",
            "seed": args.seed,
            "ratios": {"train": args.train_ratio, "valid": args.val_ratio, "test": 1.0 - args.train_ratio - args.val_ratio},
            "counts": {"train": len(train), "valid": len(valid), "test": len(test)},
            "files": {
                "train": {"path": train_path, "sha256": sha256_file(train_path)},
                "valid": {"path": valid_path, "sha256": sha256_file(valid_path)},
                "test": {"path": test_path, "sha256": sha256_file(test_path)},
            },
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(json.dumps(manifest["split"]["counts"], indent=2))


if __name__ == "__main__":
    main()
