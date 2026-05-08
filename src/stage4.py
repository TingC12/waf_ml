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
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)


TRAIN_FEATURE_CSV = "res/features/train_features.csv"
VALID_FEATURE_CSV = "res/features/valid_features.csv"
TEST_FEATURE_CSV  = "res/features/test_features.csv"

MODELS_PATH = "models"
OUTPUT_DIR = "res/features/model_results"

RANDOM_STATE = 77
MODELS = ["svc", "rf", "log_reg"]
PENALTIES = ["l1", "l2"]


def load_feature_data(path):
    df = pd.read_csv(path, encoding="utf-8-sig")

    if "label" not in df.columns:
        raise ValueError(f"{path} 缺少 label 欄位")

    y = df["label"].astype(int)

    drop_cols = [
        "payload", "label", "detected", "http_code"
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].fillna(0)

    return df, X, y, feature_cols


def get_confusion_info(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp, cm


def evaluate_model(name, model, X, y, split_name="valid"):
    y_pred = model.predict(X)

    tn, fp, fn, tp, cm = get_confusion_info(y, y_pred)

    print(f"\n=== {name} / {split_name} ===")
    print(classification_report(y, y_pred, digits=4))
    print("Confusion Matrix:")
    print(cm)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    metrics = {
        "model": name,
        "split": split_name,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": json.dumps(cm.tolist(), ensure_ascii=False),
        "classification_report": classification_report(y, y_pred, digits=4)
    }

    return metrics, y_pred


def save_metrics_csv(metrics_list, out_path):
    df = pd.DataFrame(metrics_list)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] metrics 已輸出: {out_path}")


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
    print(f"[INFO] predictions 已輸出: {out_path}")


if __name__ == "__main__":
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df, X_train, y_train, feature_cols = load_feature_data(TRAIN_FEATURE_CSV)
    valid_df, X_valid, y_valid, _ = load_feature_data(VALID_FEATURE_CSV)
    test_df,  X_test,  y_test,  _ = load_feature_data(TEST_FEATURE_CSV)

    print(f"[INFO] train size: {len(X_train)}")
    print(f"[INFO] valid size: {len(X_valid)}")
    print(f"[INFO] test size : {len(X_test)}")
    print(f"[INFO] feature count: {len(feature_cols)}")

    best_model = None
    best_name = None
    best_score = -1

    valid_metrics_records = []

    for model_name in MODELS:
        if model_name == "svc":
            for penalty in PENALTIES:
                if penalty == "l1":
                    model = LinearSVC(
                        C=0.5,
                        penalty="l1",
                        loss="squared_hinge",
                        dual=False,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        fit_intercept=False,
                        max_iter=5000
                    )
                else:
                    model = LinearSVC(
                        C=0.5,
                        penalty="l2",
                        loss="squared_hinge",
                        dual=True,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        fit_intercept=False,
                        max_iter=5000
                    )

                model.fit(X_train, y_train)
                full_name = f"LinearSVC ({penalty})"

                metrics, y_valid_pred = evaluate_model(full_name, model, X_valid, y_valid, "valid")
                valid_metrics_records.append(metrics)

                save_path = os.path.join(MODELS_PATH, f"linear_svc_rules_{penalty}.joblib")
                joblib.dump(model, save_path)

                pred_path = os.path.join(OUTPUT_DIR, f"valid_predictions_linear_svc_{penalty}.csv")
                save_predictions_csv(valid_df, y_valid, y_valid_pred, full_name, "valid", pred_path)

                if metrics["f1"] > best_score:
                    best_score = metrics["f1"]
                    best_model = model
                    best_name = full_name

        elif model_name == "rf":
            model = RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                n_estimators=300
            )
            model.fit(X_train, y_train)

            full_name = "RandomForest"
            metrics, y_valid_pred = evaluate_model(full_name, model, X_valid, y_valid, "valid")
            valid_metrics_records.append(metrics)

            save_path = os.path.join(MODELS_PATH, "rf_rules.joblib")
            joblib.dump(model, save_path)

            pred_path = os.path.join(OUTPUT_DIR, "valid_predictions_random_forest.csv")
            save_predictions_csv(valid_df, y_valid, y_valid_pred, full_name, "valid", pred_path)

            if metrics["f1"] > best_score:
                best_score = metrics["f1"]
                best_model = model
                best_name = full_name

        elif model_name == "log_reg":
            for penalty in PENALTIES:
                model = LogisticRegression(
                    C=0.5,
                    penalty=penalty,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    max_iter=3000,
                    solver="saga"
                )
                model.fit(X_train, y_train)

                full_name = f"LogisticRegression ({penalty})"
                metrics, y_valid_pred = evaluate_model(full_name, model, X_valid, y_valid, "valid")
                valid_metrics_records.append(metrics)

                save_path = os.path.join(MODELS_PATH, f"log_reg_rules_{penalty}.joblib")
                joblib.dump(model, save_path)

                pred_path = os.path.join(OUTPUT_DIR, f"valid_predictions_log_reg_{penalty}.csv")
                save_predictions_csv(valid_df, y_valid, y_valid_pred, full_name, "valid", pred_path)

                if metrics["f1"] > best_score:
                    best_score = metrics["f1"]
                    best_model = model
                    best_name = full_name

    valid_metrics_path = os.path.join(OUTPUT_DIR, "valid_metrics_summary.csv")
    save_metrics_csv(valid_metrics_records, valid_metrics_path)

    print(f"\nBest model on valid: {best_name}  F1={best_score:.4f}")

    test_metrics, y_test_pred = evaluate_model(best_name, best_model, X_test, y_test, "test")

    test_metrics_path = os.path.join(OUTPUT_DIR, "test_metrics_best_model.csv")
    save_metrics_csv([test_metrics], test_metrics_path)

    test_pred_path = os.path.join(OUTPUT_DIR, "test_predictions_best_model.csv")
    save_predictions_csv(test_df, y_test, y_test_pred, best_name, "test", test_pred_path)