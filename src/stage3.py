#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd

TRAIN_CSV = "res/splits/train.csv"
VALID_CSV = "res/splits/valid.csv"
TEST_CSV  = "res/splits/test.csv"

RULE_VOCAB_JSON = "res/features/rule_vocab.json"
OUT_DIR = "res/features"


def parse_rule_ids(rule_str):
    if pd.isna(rule_str):
        return []
    return [r.strip() for r in str(rule_str).split(",") if r.strip()]


def vectorize_rules(df, vocab, rule_col="rule_id"):
    rows = []

    for _, row in df.iterrows():
        fired = set(parse_rule_ids(row.get(rule_col, "")))
        feat = {f"rule_{rid}": int(rid in fired) for rid in vocab}

        rows.append({
            "payload": row.get("payload", ""),
            "label": row.get("label", None),
            "detected": row.get("detected", None),
            "http_code": row.get("http_code", None),
            "rule_count": row.get("rule_count", 0),
            **feat
        })

    return pd.DataFrame(rows)


def process_file(input_csv, output_csv, vocab):
    df = pd.read_csv(input_csv, encoding="utf-8-sig")

    if "rule_id" not in df.columns:
        raise ValueError(f"{input_csv} 找不到 rule_id 欄位")

    feat_df = vectorize_rules(df, vocab, rule_col="rule_id")
    feat_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"已輸出: {output_csv}  shape={feat_df.shape}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(RULE_VOCAB_JSON, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    print(f"載入 vocab，共 {len(vocab)} 條規則")

    process_file(TRAIN_CSV, os.path.join(OUT_DIR, "train_features.csv"), vocab)
    process_file(VALID_CSV, os.path.join(OUT_DIR, "valid_features.csv"), vocab)
    process_file(TEST_CSV,  os.path.join(OUT_DIR, "test_features.csv"), vocab)


if __name__ == "__main__":
    main()