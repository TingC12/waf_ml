#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import joblib
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

TRAIN_FEATURE_CSV = "res/features/train_features.csv"
VALID_FEATURE_CSV = "res/features/valid_features.csv"
TEST_FEATURE_CSV  = "res/features/test_features.csv"

MODELS_PATH = "models"
OUTPUT_DIR = "res/features/model_results"

RANDOM_STATE = 77
MODELS = ["svc", "rf", "log_reg"]
PENALTIES = ["l1", "l2"]

# 圖片裡這欄通常就是 rule feature 用 1 表示 hit
FEATURE_HIT_VALUE = 1


def load_feature_data(path):
    df = pd.read_csv(path, encoding="utf-8-sig")

    if "label" not in df.columns:
        raise ValueError(f"{path} 缺少 label 欄位")

    y = df["label"].astype(int)

    drop_cols = ["payload", "label", "detected", "http_code"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].fillna(0)

    return df, X, y, feature_cols


def get_confusion_info(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp, cm


def get_model_scores(model, X):
    """
    為了算 roc_auc：
    - 有 decision_function 用 decision_function
    - 否則用 predict_proba
    """
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    elif hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    else:
        return None


def evaluate_model_for_table(model_key, model, X_train, y_train, X_test, y_test, feature_count):
    y_pred = model.predict(X_test)
    scores = get_model_scores(model, X_test)

    tn, fp, fn, tp, cm = get_confusion_info(y_test, y_pred)

    row = {
        "model": model_key,
        "feature_hit_value": FEATURE_HIT_VALUE,
        "feature_count": int(feature_count),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores) if scores is not None else None,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": json.dumps(cm.tolist(), ensure_ascii=False)
    }

    return row, y_pred


def save_metrics_csv(metrics_list, out_path):
    df = pd.DataFrame(metrics_list)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] ML 摘要已輸出: {out_path}")


def save_predictions_csv(df_src, y_true, y_pred, model_name, split_name, out_path):
    out_df = df_src.copy().reset_index(drop=True)
    out_df["y_true"] = pd.Series(y_true).reset_index(drop=True)
    out_df["y_pred"] = pd.Series(y_pred).reset_index(drop=True)
    out_df["model"] = model_name
    out_df["split"] = split_name

    def tag_result(row):
        if row["y_true"] == 1 and row["y_pred"] == 1:
            return "TP"
        elif row["y_true"] == 0 and row["y_pred"] == 0:
            return "TN"
        elif row["y_true"] == 0 and row["y_pred"] == 1:
            return "FP"
        else:
            return "FN"

    out_df["result_type"] = out_df.apply(tag_result, axis=1)

    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 測試集預測已輸出: {out_path}")


if __name__ == "__main__":
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df, X_train, y_train, feature_cols = load_feature_data(TRAIN_FEATURE_CSV)
    valid_df, X_valid, y_valid, _ = load_feature_data(VALID_FEATURE_CSV)
    test_df,  X_test,  y_test,  _ = load_feature_data(TEST_FEATURE_CSV)

    print("\n===== Stage 3: Training ML models on CRS rule vectors =====")

    results = []
    prediction_records = []

    # -----------------------------
    # Linear SVM
    # -----------------------------
    print("[ML] Training linear_svm ...")
    svm_model = LinearSVC(
        C=0.5,
        penalty="l2",
        loss="squared_hinge",
        dual=True,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        fit_intercept=False,
        max_iter=10000
    )
    svm_model.fit(X_train, y_train)
    svm_path = os.path.join(MODELS_PATH, "linear_svm_rules.joblib")
    joblib.dump(svm_model, svm_path)

    svm_row, svm_pred = evaluate_model_for_table(
        model_key="linear_svm",
        model=svm_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_count=len(feature_cols)
    )
    results.append(svm_row)
    prediction_records.append(("linear_svm", svm_pred))

    # -----------------------------
    # Random Forest
    # -----------------------------
    print("[ML] Training random_forest ...")
    rf_model = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        n_estimators=300
    )
    rf_model.fit(X_train, y_train)
    rf_path = os.path.join(MODELS_PATH, "random_forest_rules.joblib")
    joblib.dump(rf_model, rf_path)

    rf_row, rf_pred = evaluate_model_for_table(
        model_key="random_forest",
        model=rf_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_count=len(feature_cols)
    )
    results.append(rf_row)
    prediction_records.append(("random_forest", rf_pred))

    # -----------------------------
    # Logistic Regression
    # -----------------------------
    print("[ML] Training logistic_regression ...")
    lr_model = LogisticRegression(
        C=0.5,
        penalty="l2",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_iter=5000,
        solver="saga"
    )
    lr_model.fit(X_train, y_train)
    lr_path = os.path.join(MODELS_PATH, "logistic_regression_rules.joblib")
    joblib.dump(lr_model, lr_path)

    lr_row, lr_pred = evaluate_model_for_table(
        model_key="logistic_regression",
        model=lr_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_count=len(feature_cols)
    )
    results.append(lr_row)
    prediction_records.append(("logistic_regression", lr_pred))

    # -----------------------------
    # 輸出 summary CSV
    # -----------------------------
    summary_df = pd.DataFrame(results)

    # 圖片風格：照 f1 由高到低排
    summary_df = summary_df.sort_values(by="f1", ascending=False).reset_index(drop=True)

    summary_path = os.path.join(OUTPUT_DIR, "ml_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] 使用規則數: {len(feature_cols)}")
    print(f"[INFO] ML 摘要已輸出: {summary_path}")

    # -----------------------------
    # 輸出所有模型測試集預測
    # -----------------------------
    all_pred_rows = []
    for model_name, y_pred in prediction_records:
        tmp_df = test_df.copy().reset_index(drop=True)
        tmp_df["y_true"] = y_test.reset_index(drop=True)
        tmp_df["y_pred"] = pd.Series(y_pred)
        tmp_df["model"] = model_name
        all_pred_rows.append(tmp_df)

    pred_df = pd.concat(all_pred_rows, ignore_index=True)
    pred_path = os.path.join(OUTPUT_DIR, "ml_predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 測試集預測已輸出: {pred_path}")

    # -----------------------------
    # 終端顯示成圖片那種表格格式
    # -----------------------------
    show_cols = [
        "model",
        "feature_hit_value",
        "feature_count",
        "train_size",
        "test_size",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", None)

    print("\n" + summary_df[show_cols].to_string(index=False))