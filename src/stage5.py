#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "models/log_reg_rules_l2.joblib"
FEATURE_CSV = "res/features/train_features.csv"

OUT_FLOAT_CSV = "res/features/rule_weights_for_waf_float.csv"
OUT_INT_CSV   = "res/features/rule_weights_for_waf_int.csv"

# =========================
# Mapping Config
# =========================
MAX_WAF_SCORE = 10              # 建議 10 或 20，不要 5
NEGATIVE_MODE = "shift"         # "zero" or "shift"
APPLY_SQRT = True               # True: 拉開小權重差距
ROUND_TO_INT = True             # True: 另外輸出整數版給 WAF
DROP_RULE_COUNT = True          # True: 不把 rule_count 當成 rule 權重
MIN_NONZERO_SCORE = 1           # 若 >0 但 round 後變 0，可補成 1


def load_feature_columns(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    drop_cols = ["payload", "label", "detected", "http_code"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return feature_cols


def map_weights(raw_weights, max_waf_score=10, negative_mode="shift", apply_sqrt=True):
    w = np.array(raw_weights, dtype=float)

    if negative_mode == "zero":
        w[w < 0] = 0.0
        if np.allclose(w.max(), 0.0):
            mapped = np.zeros_like(w, dtype=float)
        else:
            mapped = w / w.max()

    elif negative_mode == "shift":
        w_min = w.min()
        w_max = w.max()
        if np.allclose(w_max, w_min):
            mapped = np.ones_like(w, dtype=float) * 0.5
        else:
            mapped = (w - w_min) / (w_max - w_min)

    else:
        raise ValueError("NEGATIVE_MODE 只能是 'zero' 或 'shift'")

    if apply_sqrt:
        mapped = np.sqrt(mapped)

    mapped = mapped * max_waf_score
    return mapped


def safe_rule_id(feature_name):
    if feature_name.startswith("rule_"):
        return feature_name.replace("rule_", "")
    return feature_name


def make_integer_scores(float_scores, min_nonzero_score=1):
    int_scores = np.rint(float_scores).astype(int)

    # 避免有些 >0 的小權重 round 後直接變 0
    mask = (float_scores > 0) & (int_scores == 0)
    int_scores[mask] = min_nonzero_score

    return int_scores


def main():
    os.makedirs(os.path.dirname(OUT_FLOAT_CSV), exist_ok=True)

    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "coef_"):
        raise ValueError("此模型沒有 coef_，請用 LogisticRegression 或 LinearSVC")

    feature_cols = load_feature_columns(FEATURE_CSV)
    raw_weights = model.coef_.ravel()

    if len(raw_weights) != len(feature_cols):
        raise ValueError(
            f"模型權重數量({len(raw_weights)})與 feature 數量({len(feature_cols)})不一致"
        )

    df = pd.DataFrame({
        "feature": feature_cols,
        "rule_id": [safe_rule_id(c) for c in feature_cols],
        "raw_weight": raw_weights
    })

    if DROP_RULE_COUNT:
        df = df[df["feature"] != "rule_count"].copy()

    mapped_float = map_weights(
        raw_weights=df["raw_weight"].values,
        max_waf_score=MAX_WAF_SCORE,
        negative_mode=NEGATIVE_MODE,
        apply_sqrt=APPLY_SQRT
    )

    df["mapped_weight_float"] = mapped_float

    if ROUND_TO_INT:
        df["mapped_weight_int"] = make_integer_scores(
            df["mapped_weight_float"].values,
            min_nonzero_score=MIN_NONZERO_SCORE
        )
    else:
        df["mapped_weight_int"] = np.nan

    df = df.sort_values(
        ["mapped_weight_float", "raw_weight"],
        ascending=[False, False]
    ).reset_index(drop=True)

    df.to_csv(OUT_FLOAT_CSV, index=False, encoding="utf-8-sig")

    if ROUND_TO_INT:
        df_int = df[["feature", "rule_id", "raw_weight", "mapped_weight_int"]].copy()
        df_int = df_int.rename(columns={"mapped_weight_int": "mapped_weight"})
        df_int.to_csv(OUT_INT_CSV, index=False, encoding="utf-8-sig")

    print(f"[DONE] saved float csv: {OUT_FLOAT_CSV}")
    if ROUND_TO_INT:
        print(f"[DONE] saved int csv  : {OUT_INT_CSV}")

    print("\n=== Top 20 rules ===")
    print(df.head(20).to_string(index=False))

    print("\n=== Mapping Config ===")
    print(f"MODEL_PATH       : {MODEL_PATH}")
    print(f"FEATURE_CSV      : {FEATURE_CSV}")
    print(f"MAX_WAF_SCORE    : {MAX_WAF_SCORE}")
    print(f"NEGATIVE_MODE    : {NEGATIVE_MODE}")
    print(f"APPLY_SQRT       : {APPLY_SQRT}")
    print(f"ROUND_TO_INT     : {ROUND_TO_INT}")
    print(f"DROP_RULE_COUNT  : {DROP_RULE_COUNT}")
    print(f"MIN_NONZERO_SCORE: {MIN_NONZERO_SCORE}")


if __name__ == "__main__":
    main()