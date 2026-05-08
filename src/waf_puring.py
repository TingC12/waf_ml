#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# =========================
# Config
# =========================
TRAIN_FEATURE_CSV = "res/features/train_features.csv"
VALID_FEATURE_CSV = "res/features/valid_features.csv"
TEST_FEATURE_CSV  = "res/features/test_features.csv"

RULE_VOCAB_JSON = "res/features/rule_vocab.json"

OUTPUT_DIR = "res/pruning"
MODEL_DIR = "models"

RANDOM_STATE = 77
MAX_ITER = 2000

# pruning mode:
#   "threshold" -> 根據 |weight| > threshold 保留
#   "topk"      -> 保留前 k 個最大權重
PRUNING_MODE = "threshold"

# threshold mode 用
THRESHOLDS = [0.1, 0.2, 0.3, 0.5, 1.0]
# topk mode 用
TOPK_LIST = [5, 10, 15, 20, 25, 30]

# 若你想只跑 valid 不跑 test，可以改 False
RUN_TEST = True


# =========================
# Utils
# =========================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_feature_data(path):
    df = pd.read_csv(path, encoding="utf-8-sig")

    y = df["label"].astype(int).values

    drop_cols = ["payload", "label", "detected", "http_code"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values
    return X, y, feature_cols, df


def load_rule_vocab(rule_vocab_path):
    if not os.path.exists(rule_vocab_path):
        return None

    with open(rule_vocab_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # 常見情況1: list -> ["941100", "941110", ...]
    if isinstance(obj, list):
        return [str(x) for x in obj]

    # 常見情況2: dict -> {"941100": 0, "941110": 1, ...}
    if isinstance(obj, dict):
        # 轉成 index 對應 rule_id
        if all(isinstance(v, int) for v in obj.values()):
            inv = sorted(obj.items(), key=lambda x: x[1])
            return [str(rule_id) for rule_id, _ in inv]

    return None


def feature_cols_to_rule_ids(feature_cols):
    """
    feature name 可能是:
      - rule_941100
      - 941100
    這裡統一轉成 rule_id 字串
    """
    rule_ids = []
    for c in feature_cols:
        c = str(c)
        if c.startswith("rule_"):
            rule_ids.append(c.replace("rule_", "", 1))
        else:
            rule_ids.append(c)
    return rule_ids


def train_lr(X_train, y_train):
    model = LogisticRegression(
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        solver="liblinear"
    )
    model.fit(X_train, y_train)
    return model


def calc_metrics(y_true, y_pred, y_score=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": fp / (fp + tn + 1e-9),
        "tpr": tp / (tp + fn + 1e-9),
        "bypass_rate": fn / (tp + fn + 1e-9),
        "specificity": tn / (tn + fp + 1e-9),
        "confusion_matrix": str([[int(tn), int(fp)], [int(fn), int(tp)]])
    }

    if y_score is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        except Exception:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan

    return metrics


def evaluate_model(model, X, y):
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X)
    else:
        y_score = None

    return calc_metrics(y, y_pred, y_score), y_pred


def save_rule_weights_csv(rule_ids, weights, out_csv):
    df = pd.DataFrame({
        "rule_id": rule_ids,
        "weight": weights,
        "abs_weight": np.abs(weights)
    }).sort_values("abs_weight", ascending=False)

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df


def select_rules_by_threshold(weights, threshold):
    keep_idx = np.where(np.abs(weights) > threshold)[0]
    return keep_idx


def select_rules_by_topk(weights, k):
    k = min(k, len(weights))
    keep_idx = np.argsort(np.abs(weights))[-k:]
    keep_idx = np.sort(keep_idx)
    return keep_idx


def run_one_experiment(
    exp_name,
    keep_idx,
    X_train, y_train,
    X_valid, y_valid,
    X_test, y_test,
    rule_ids
):
    X_train_r = X_train[:, keep_idx]
    X_valid_r = X_valid[:, keep_idx]
    X_test_r  = X_test[:, keep_idx] if X_test is not None else None

    model = train_lr(X_train_r, y_train)

    valid_metrics, valid_pred = evaluate_model(model, X_valid_r, y_valid)
    valid_metrics["split"] = "valid"
    valid_metrics["experiment"] = exp_name
    valid_metrics["rule_count"] = len(keep_idx)

    results = [valid_metrics]

    test_pred = None
    if X_test is not None and y_test is not None:
        test_metrics, test_pred = evaluate_model(model, X_test_r, y_test)
        test_metrics["split"] = "test"
        test_metrics["experiment"] = exp_name
        test_metrics["rule_count"] = len(keep_idx)
        results.append(test_metrics)

    kept_rules = [rule_ids[i] for i in keep_idx]

    return {
        "model": model,
        "results": results,
        "keep_idx": keep_idx,
        "kept_rules": kept_rules,
        "valid_pred": valid_pred,
        "test_pred": test_pred
    }


# =========================
# Main
# =========================
def main():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(MODEL_DIR)

    print("Loading features...")
    X_train, y_train, feature_cols_train, train_df = load_feature_data(TRAIN_FEATURE_CSV)
    X_valid, y_valid, feature_cols_valid, valid_df = load_feature_data(VALID_FEATURE_CSV)

    if RUN_TEST and os.path.exists(TEST_FEATURE_CSV):
        X_test, y_test, feature_cols_test, test_df = load_feature_data(TEST_FEATURE_CSV)
    else:
        X_test, y_test, feature_cols_test, test_df = None, None, None, None

    # 特徵欄位一致性檢查
    if feature_cols_train != feature_cols_valid:
        raise ValueError("train/valid 的 feature columns 不一致")
    if X_test is not None and feature_cols_train != feature_cols_test:
        raise ValueError("train/test 的 feature columns 不一致")

    # rule_id 對應
    rule_vocab = load_rule_vocab(RULE_VOCAB_JSON)
    if rule_vocab is not None and len(rule_vocab) == len(feature_cols_train):
        rule_ids = rule_vocab
    else:
        rule_ids = feature_cols_to_rule_ids(feature_cols_train)

    print(f"Total rules/features: {len(rule_ids)}")

    # ========= baseline =========
    print("\nTraining baseline model...")
    baseline_model = train_lr(X_train, y_train)
    joblib.dump(baseline_model, os.path.join(MODEL_DIR, "logistic_regression_baseline.joblib"))

    baseline_valid_metrics, baseline_valid_pred = evaluate_model(baseline_model, X_valid, y_valid)
    baseline_valid_metrics["split"] = "valid"
    baseline_valid_metrics["experiment"] = "baseline_all_rules"
    baseline_valid_metrics["rule_count"] = len(rule_ids)

    all_results = [baseline_valid_metrics]

    baseline_test_pred = None
    if X_test is not None:
        baseline_test_metrics, baseline_test_pred = evaluate_model(baseline_model, X_test, y_test)
        baseline_test_metrics["split"] = "test"
        baseline_test_metrics["experiment"] = "baseline_all_rules"
        baseline_test_metrics["rule_count"] = len(rule_ids)
        all_results.append(baseline_test_metrics)

    baseline_weights = baseline_model.coef_[0]

    weight_df = save_rule_weights_csv(
        rule_ids=rule_ids,
        weights=baseline_weights,
        out_csv=os.path.join(OUTPUT_DIR, "baseline_rule_weights.csv")
    )

    # ========= pruning =========
    pruning_records = []

    if PRUNING_MODE == "threshold":
        search_values = THRESHOLDS
        print("\nRunning threshold pruning...")
        for th in search_values:
            keep_idx = select_rules_by_threshold(baseline_weights, th)

            if len(keep_idx) == 0:
                continue

            exp_name = f"threshold_{th}"
            print(f"  {exp_name}: keep {len(keep_idx)} rules")

            exp = run_one_experiment(
                exp_name=exp_name,
                keep_idx=keep_idx,
                X_train=X_train, y_train=y_train,
                X_valid=X_valid, y_valid=y_valid,
                X_test=X_test, y_test=y_test,
                rule_ids=rule_ids
            )

            all_results.extend(exp["results"])

            pruning_records.append({
                "experiment": exp_name,
                "mode": "threshold",
                "value": th,
                "rule_count": len(exp["kept_rules"]),
                "kept_rules": ",".join(exp["kept_rules"])
            })

            # 存模型
            joblib.dump(
                exp["model"],
                os.path.join(MODEL_DIR, f"logistic_regression_{exp_name}.joblib")
            )

            # 存保留規則
            pd.DataFrame({
                "rule_id": exp["kept_rules"]
            }).to_csv(
                os.path.join(OUTPUT_DIR, f"kept_rules_{exp_name}.csv"),
                index=False,
                encoding="utf-8-sig"
            )

    elif PRUNING_MODE == "topk":
        search_values = TOPK_LIST
        print("\nRunning top-k pruning...")
        for k in search_values:
            keep_idx = select_rules_by_topk(baseline_weights, k)

            if len(keep_idx) == 0:
                continue

            exp_name = f"topk_{k}"
            print(f"  {exp_name}: keep {len(keep_idx)} rules")

            exp = run_one_experiment(
                exp_name=exp_name,
                keep_idx=keep_idx,
                X_train=X_train, y_train=y_train,
                X_valid=X_valid, y_valid=y_valid,
                X_test=X_test, y_test=y_test,
                rule_ids=rule_ids
            )

            all_results.extend(exp["results"])

            pruning_records.append({
                "experiment": exp_name,
                "mode": "topk",
                "value": k,
                "rule_count": len(exp["kept_rules"]),
                "kept_rules": ",".join(exp["kept_rules"])
            })

            joblib.dump(
                exp["model"],
                os.path.join(MODEL_DIR, f"logistic_regression_{exp_name}.joblib")
            )

            pd.DataFrame({
                "rule_id": exp["kept_rules"]
            }).to_csv(
                os.path.join(OUTPUT_DIR, f"kept_rules_{exp_name}.csv"),
                index=False,
                encoding="utf-8-sig"
            )

    else:
        raise ValueError("PRUNING_MODE 必須是 'threshold' 或 'topk'")

    # ========= save summary =========
    results_df = pd.DataFrame(all_results)
    results_df = results_df[
        [
            "experiment", "split", "rule_count",
            "accuracy", "precision", "recall", "f1", "roc_auc",
            "tn", "fp", "fn", "tp",
            "fpr", "tpr", "bypass_rate", "specificity",
            "confusion_matrix"
        ]
    ]
    results_df.to_csv(
        os.path.join(OUTPUT_DIR, "pruning_experiment_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    pruning_df = pd.DataFrame(pruning_records)
    pruning_df.to_csv(
        os.path.join(OUTPUT_DIR, "pruning_rule_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # ========= choose best on valid =========
    valid_only = results_df[results_df["split"] == "valid"].copy()
    valid_only = valid_only.sort_values(
        by=["f1", "accuracy", "precision", "recall"],
        ascending=False
    )

    best_row = valid_only.iloc[0]
    best_exp = best_row["experiment"]

    print("\n==============================")
    print("Best experiment on VALID")
    print("==============================")
    print(best_row.to_string())

    # 存最佳摘要
    pd.DataFrame([best_row]).to_csv(
        os.path.join(OUTPUT_DIR, "best_valid_experiment.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # ========= save predictions =========
    # baseline valid predictions
    valid_pred_df = valid_df.copy()
    valid_pred_df["baseline_pred"] = baseline_valid_pred
    valid_pred_df.to_csv(
        os.path.join(OUTPUT_DIR, "valid_predictions_baseline.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    if X_test is not None:
        test_pred_df = test_df.copy()
        test_pred_df["baseline_pred"] = baseline_test_pred
        test_pred_df.to_csv(
            os.path.join(OUTPUT_DIR, "test_predictions_baseline.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    print("\nSaved files:")
    print(f"- {os.path.join(OUTPUT_DIR, 'baseline_rule_weights.csv')}")
    print(f"- {os.path.join(OUTPUT_DIR, 'pruning_experiment_results.csv')}")
    print(f"- {os.path.join(OUTPUT_DIR, 'pruning_rule_summary.csv')}")
    print(f"- {os.path.join(OUTPUT_DIR, 'best_valid_experiment.csv')}")
    print("\nDone.")


if __name__ == "__main__":
    main()