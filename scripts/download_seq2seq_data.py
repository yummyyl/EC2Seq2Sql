import argparse
import hashlib
import os
import sys
import urllib.request


COMMIT_SHA = "62d54af08ba52e8196e664fcec01122a4d4e38ab"
RAW_URL = (
    "https://raw.githubusercontent.com/uw-bionlp/clinical-trials-gov-data/"
    f"{COMMIT_SHA}/data/seq2seq/train.json"
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=str,
        default="data/processed/ec2dsl_pairs.json",
        help="Output path for the downloaded dataset JSON file.",
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"Downloading dataset from:\n  {RAW_URL}\nSaving to:\n  {args.out}")

    try:
        urllib.request.urlretrieve(RAW_URL, args.out)
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)

    print("Download finished.")
    print("SHA256:", sha256_file(args.out))
    print("Commit:", COMMIT_SHA)


if __name__ == "__main__":
    main()
