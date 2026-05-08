#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd

INPUT_CSV = "res/splits/train.csv"
OUT_RULE_MAP_JSON = "res/features/rule_vocab.json"

def parse_rule_ids(rule_str):
    if pd.isna(rule_str):
        return []
    return [r.strip() for r in str(rule_str).split(",") if r.strip()]

def build_rule_vocab(df, rule_col="rule_id"):
    all_rules = set()
    for v in df[rule_col]:
        all_rules.update(parse_rule_ids(v))
    return sorted(all_rules)

def main():
    os.makedirs("res/features", exist_ok=True)

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    if "rule_id" not in df.columns:
        raise ValueError("找不到 rule_id 欄位")

    vocab = build_rule_vocab(df, rule_col="rule_id")
    print(f"總共收集到 {len(vocab)} 條規則特徵")

    with open(OUT_RULE_MAP_JSON, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"規則字典已輸出: {OUT_RULE_MAP_JSON}")

if __name__ == "__main__":
    main()