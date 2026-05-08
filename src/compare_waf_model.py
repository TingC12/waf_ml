#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time as t
import requests
import pandas as pd
import subprocess
import joblib
from datetime import datetime

# =========================
# Config
# =========================
INPUT_FILE = "res/dataset/XSS_dataset.csv"   # 要有 payload,label
OUTPUT_DIR = "res/predict"
MODEL_PATH = "models/linear_svc_rules_l2.joblib"
RULE_VOCAB_JSON = "res/features/rule_vocab.json"

DOCKER_CONTAINER = "waf"
MODSEC_LOG_PATH = "/tmp/modsec_audit.log"
POST_URL = "http://localhost:80/comment"

SLEEP_AFTER_POST = 0.15
LOG_WRITE_WAIT = 0.05
MAX_WAIT_ROUNDS = 15
REQUEST_TIMEOUT = 5

BLOCK_HTTP_CODES = {403, 406, 429, 500, 501, 502, 503}
TOTAL_SCORE_REGEX_JSON = re.compile(r"Total\s*Score:\s*(\d+)", re.IGNORECASE)


# =========================
# 基本工具
# =========================
def run_cmd(cmd):
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=True
        )
        return res.stdout.strip()
    except Exception:
        return ""


def check_container_running(container_name: str) -> bool:
    out = run_cmd(["docker", "ps", "--filter", f"name=^{container_name}$", "--format", "{{.Names}}"])
    return container_name in out.splitlines()


def wait_for_waf_ready(timeout_sec: int = 30) -> bool:
    start = t.time()
    while t.time() - start < timeout_sec:
        try:
            resp = requests.get("http://localhost:80", timeout=3)
            if resp.status_code < 500:
                return True
        except Exception:
            pass
        t.sleep(1)
    return False


# =========================
# 讀取 payload + label
# =========================
def normalize_label(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"1", "xss", "attack", "malicious", "true", "yes"}:
        return 1
    if s in {"0", "benign", "normal", "clean", "false", "no"}:
        return 0
    return None


def load_payloads(filepath):
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".txt":
        with open(filepath, encoding="utf-8") as f:
            payloads = [line.strip() for line in f if line.strip()]
        df = pd.DataFrame({
            "payload": payloads,
            "label": [None] * len(payloads)
        })

    elif ext == ".csv":
        df_raw = pd.read_csv(filepath, encoding="utf-8-sig")

        if len(df_raw.columns) == 1:
            df_raw = pd.read_csv(filepath, header=None, encoding="utf-8-sig")

        if isinstance(df_raw.columns[0], int):
            df = pd.DataFrame({
                "payload": df_raw.iloc[:, 0].astype(str).str.strip()
            })
            if df_raw.shape[1] >= 2:
                df["label"] = df_raw.iloc[:, 1]
            else:
                df["label"] = None
        else:
            cols_lower = {c.lower(): c for c in df_raw.columns}

            payload_col = None
            for c in ["payload", "message", "text", "input"]:
                if c in cols_lower:
                    payload_col = cols_lower[c]
                    break
            if payload_col is None:
                payload_col = df_raw.columns[0]

            label_col = None
            for c in ["label", "y", "target", "class"]:
                if c in cols_lower:
                    label_col = cols_lower[c]
                    break

            df = pd.DataFrame({
                "payload": df_raw[payload_col].astype(str).str.strip()
            })

            if label_col is not None:
                df["label"] = df_raw[label_col]
            else:
                df["label"] = None
    else:
        raise ValueError(f"不支援的檔案格式: {ext}")

    df = df[df["payload"].astype(str).str.strip() != ""].copy()
    df["label"] = df["label"].apply(normalize_label)
    df.reset_index(drop=True, inplace=True)

    print(f"✅ 已載入 {len(df)} 筆 payload")
    print(f"✅ 有標註 label 的樣本數: {df['label'].notna().sum()}")
    return df


# =========================
# Docker audit log
# =========================
def audit_log_exists(container_name: str, log_path: str) -> bool:
    try:
        res = subprocess.run(
            ["docker", "exec", container_name, "sh", "-c", f"test -f {log_path} && echo YES || echo NO"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=True
        )
        return res.stdout.strip() == "YES"
    except Exception:
        return False


def read_docker_audit_log(container_name: str, log_path: str) -> str:
    try:
        res = subprocess.run(
            ["docker", "exec", container_name, "cat", log_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=True
        )
        return res.stdout or ""
    except Exception:
        return ""


# =========================
# 解析 JSON log
# =========================
def parse_all_modsec_json_lines(audit_text: str):
    objs = []
    for line in audit_text.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            objs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return objs


def extract_tx_info(obj):
    tx = obj.get("transaction", {}) or {}
    uid = tx.get("unique_id", "") or tx.get("id", "") or ""

    resp = tx.get("response", {}) or obj.get("response", {}) or {}
    http_code = resp.get("http_code") or resp.get("status")

    msgs = obj.get("messages", [])
    if not msgs and isinstance(tx, dict):
        msgs = tx.get("messages", [])
    if not msgs:
        audit = obj.get("audit_data", {}) or obj.get("auditData", {}) or {}
        msgs = audit.get("messages", [])

    rule_ids = []
    total_score = 0

    for m in msgs or []:
        if isinstance(m, dict):
            msg = m.get("message") or m.get("msg") or ""
            details = m.get("details", {}) or {}
            rid = details.get("ruleId") or details.get("rule_id")
            if rid:
                rule_ids.append(str(rid))
            if msg:
                m2 = TOTAL_SCORE_REGEX_JSON.search(msg)
                if m2:
                    total_score = int(m2.group(1))
        elif isinstance(m, str):
            m2 = re.search(r'(?i)rule(?:Id|Id:)\s*"?(\d+)"?', m)
            if m2:
                rule_ids.append(m2.group(1))
            m3 = TOTAL_SCORE_REGEX_JSON.search(m)
            if m3:
                total_score = int(m3.group(1))

    return {
        "uid": uid,
        "http_code": http_code,
        "rule_ids": rule_ids,
        "total_score": total_score
    }


def get_all_uids(audit_text: str):
    objs = parse_all_modsec_json_lines(audit_text)
    uids = []
    for obj in objs:
        info = extract_tx_info(obj)
        if info["uid"]:
            uids.append(info["uid"])
    return uids


# =========================
# 判斷 WAF
# =========================
def is_blocked(http_code, rule_ids, total_score):
    if http_code in BLOCK_HTTP_CODES:
        return 1
    if len(rule_ids) > 0 and total_score > 0:
        return 1
    return 0


def post_and_check_modsecurity(payload: str):
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"message": payload}

    audit_before = read_docker_audit_log(DOCKER_CONTAINER, MODSEC_LOG_PATH)
    seen_uids = set(get_all_uids(audit_before))

    http_code = None

    try:
        resp = requests.post(
            POST_URL,
            headers=headers,
            data=data,
            timeout=REQUEST_TIMEOUT
        )
        http_code = resp.status_code
    except Exception as e:
        print(f"[POST ERROR] {e}")
        http_code = None

    matched_info = None

    for _ in range(MAX_WAIT_ROUNDS):
        t.sleep(SLEEP_AFTER_POST + LOG_WRITE_WAIT)
        audit_after = read_docker_audit_log(DOCKER_CONTAINER, MODSEC_LOG_PATH)
        objs = parse_all_modsec_json_lines(audit_after)

        for obj in reversed(objs):
            info = extract_tx_info(obj)
            uid = info["uid"]
            if uid and uid not in seen_uids:
                matched_info = info
                break

        if matched_info is not None:
            break

    rule_ids = []
    total_score = 0

    if matched_info is not None:
        rule_ids = matched_info["rule_ids"]
        total_score = matched_info["total_score"]
        if matched_info["http_code"] is not None:
            http_code = matched_info["http_code"]

    detected = is_blocked(http_code, rule_ids, total_score)
    waf_result = "block" if detected == 1 else "bypass"

    return {
        "pred": detected,
        "waf_result": waf_result,
        "rule_ids": rule_ids
    }


# =========================
# 特徵向量
# =========================
def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


def vectorize_for_model(rule_ids, vocab):
    fired = set(str(r).strip() for r in rule_ids if str(r).strip())
    row = {"rule_count": len(fired)}
    for rid in vocab:
        row[f"rule_{rid}"] = int(rid in fired)
    return row


# =========================
# 評估指標
# =========================
def calc_metrics(y_true, y_pred):
    y_true = pd.Series(y_true).astype(int)
    y_pred = pd.Series(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
        "TNR": tnr
    }


def print_metrics(title, metrics):
    print(f"\n=== {title} ===")
    print(f"TP: {metrics['TP']}")
    print(f"TN: {metrics['TN']}")
    print(f"FP: {metrics['FP']}")
    print(f"FN: {metrics['FN']}")
    print(f"Accuracy : {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall   : {metrics['Recall']:.4f}")
    print(f"F1-score : {metrics['F1']:.4f}")
    print(f"FPR      : {metrics['FPR']:.4f}")
    print(f"TNR      : {metrics['TNR']:.4f}")


# =========================
# 主流程
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not check_container_running(DOCKER_CONTAINER):
        raise RuntimeError(f"找不到正在執行的容器: {DOCKER_CONTAINER}")

    if not wait_for_waf_ready():
        raise RuntimeError(f"WAF 服務尚未就緒: {POST_URL}")

    if not audit_log_exists(DOCKER_CONTAINER, MODSEC_LOG_PATH):
        raise RuntimeError(f"找不到 audit log: {MODSEC_LOG_PATH}")

    vocab = load_vocab(RULE_VOCAB_JSON)
    model = joblib.load(MODEL_PATH)

    print(f"[INFO] 載入 vocab: {len(vocab)} rules")
    print(f"[INFO] 載入模型: {MODEL_PATH}")

    data_df = load_payloads(INPUT_FILE)
    rows = []

    for i, row in data_df.iterrows():
        payload = row["payload"]
        label = row["label"]

        waf_info = post_and_check_modsecurity(payload)
        feature_row = vectorize_for_model(waf_info["rule_ids"], vocab)

        ordered_cols = ["rule_count"] + [f"rule_{rid}" for rid in vocab]
        X_one = pd.DataFrame([feature_row]).reindex(columns=ordered_cols, fill_value=0)
        ml_pred = int(model.predict(X_one)[0])

        rows.append({
            "index": i + 1,
            "payload": payload,
            "label": label,
            "waf_pred": waf_info["pred"],
            "waf_result": waf_info["waf_result"],
            "rule_count": len(waf_info["rule_ids"]),
            "rule_id": ",".join(waf_info["rule_ids"]),
            "ml_prediction": ml_pred
        })

        print(
            f"[{i+1}/{len(data_df)}] "
            f"label={label} "
            f"waf={waf_info['pred']} "
            f"ml={ml_pred} "
            f"rules={len(waf_info['rule_ids'])}"
        )

    result_df = pd.DataFrame(rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(OUTPUT_DIR, f"waf_ml_eval_{timestamp}.csv")
    result_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ 結果已輸出: {out_csv}")

    labeled_df = result_df[result_df["label"].notna()].copy()

    if labeled_df.empty:
        print("\n沒有 label，無法計算準確率。")
        return

    waf_metrics = calc_metrics(labeled_df["label"], labeled_df["waf_pred"])
    ml_metrics = calc_metrics(labeled_df["label"], labeled_df["ml_prediction"])

    print_metrics("WAF Metrics", waf_metrics)
    print_metrics("ML Metrics", ml_metrics)

    summary_path = os.path.join(OUTPUT_DIR, f"waf_ml_metrics_{timestamp}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for title, metrics in [("WAF Metrics", waf_metrics), ("ML Metrics", ml_metrics)]:
            f.write(f"=== {title} ===\n")
            for k, v in metrics.items():
                if isinstance(v, float):
                    f.write(f"{k}: {v:.4f}\n")
                else:
                    f.write(f"{k}: {v}\n")
            f.write("\n")

    print(f"\n✅ 指標摘要已輸出: {summary_path}")


if __name__ == "__main__":
    main()