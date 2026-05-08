#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_CSV = "res/stats/waf_eval_results_20260317_235436.csv"   # 改成你的檔名
OUT_DIR = "res/splits"
RANDOM_STATE = 77


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    if "label" not in df.columns:
        raise ValueError("找不到 label 欄位")

    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)

    # 先切 test 15%
    train_valid_df, test_df = train_test_split(
        df,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    # 再從剩下的 85% 切出 valid，最後約為 70/15/15
    train_df, valid_df = train_test_split(
        train_valid_df,
        test_size=0.17647,
        random_state=RANDOM_STATE,
        stratify=train_valid_df["label"]
    )

    train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False, encoding="utf-8-sig")
    valid_df.to_csv(os.path.join(OUT_DIR, "valid.csv"), index=False, encoding="utf-8-sig")
    test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False, encoding="utf-8-sig")

    print("切分完成")
    print(f"train: {len(train_df)}")
    print(f"valid: {len(valid_df)}")
    print(f"test : {len(test_df)}")


if __name__ == "__main__":
    main()