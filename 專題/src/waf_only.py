#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time as t
import asyncio
import requests
import pandas as pd
import subprocess
from datetime import datetime

# ---------------- Config ----------------
INPUT_FILE = "./res/splits/test.csv"

DOCKER_CONTAINER = "waf"
MODSEC_LOG_PATH = "/tmp/modsec_audit.log"
POST_URL = "http://localhost:80/comment"

SLEEP_AFTER_POST = 0.15
LOG_WRITE_WAIT = 0.05
MAX_WAIT_ROUNDS = 15
REQUEST_TIMEOUT = 5

BLOCK_HTTP_CODES = {403, 406, 429, 500, 501, 502, 503}

TOTAL_SCORE_REGEX_JSON = re.compile(r"Total\s*Score:\s*(\d+)", re.IGNORECASE)


# ---------------- 基本工具 ----------------
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
    except subprocess.CalledProcessError as e:
        print(f"[CMD ERROR] {' '.join(cmd)}")
        print(e.stderr)
        return ""
    except Exception as e:
        print(f"[CMD EXCEPTION] {' '.join(cmd)} -> {e}")
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


# ---------------- 讀取 payload ----------------
def load_payloads(filepath):
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".txt":
        print("[INFO] 偵測到 TXT 格式，逐行讀取...")
        with open(filepath, encoding="utf-8") as f:
            payloads = [line.strip() for line in f if line.strip()]
        df = pd.DataFrame({
            "payload": payloads,
            "label": [None] * len(payloads)
        })

    elif ext == ".csv":
        print("[INFO] 偵測到 CSV 格式，嘗試辨識欄位...")
        df_raw = pd.read_csv(filepath, encoding="utf-8-sig")

        if len(df_raw.columns) == 1 and df_raw.columns[0] not in ["payload", "message", "text", "input", "label", "y", "target", "class"]:
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

            label_col = None
            for c in ["label", "y", "target", "class"]:
                if c in cols_lower:
                    label_col = cols_lower[c]
                    break

            if payload_col is None:
                payload_col = df_raw.columns[0]

            df = pd.DataFrame({
                "payload": df_raw[payload_col].astype(str).str.strip()
            })

            if label_col is not None:
                df["label"] = df_raw[label_col]
            else:
                df["label"] = None

        df = df[df["payload"].astype(str).str.strip() != ""].copy()
        df.reset_index(drop=True, inplace=True)

    else:
        raise ValueError(f"不支援的檔案格式: {ext}，請使用 .txt 或 .csv")

    def normalize_label(x):
        if pd.isna(x):
            return None
        s = str(x).strip().lower()
        if s in {"1", "xss", "attack", "malicious", "true", "yes"}:
            return 1
        if s in {"0", "benign", "normal", "clean", "false", "no"}:
            return 0
        return None

    df["label"] = df["label"].apply(normalize_label)

    label_count = df["label"].notna().sum()
    print(f"✅ 已載入 {len(df)} 筆 payload")
    print(f"✅ 其中有標註 label 的樣本數: {label_count}")

    return df


# ---------------- 讀取 Docker audit log ----------------
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
    except Exception as e:
        print(f"[read_docker_audit_log] Failed: {e}")
        return ""


# ---------------- 解析 JSON log ----------------
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

    rule_ids, messages, refs, total_score = [], [], [], 0

    for m in msgs or []:
        if isinstance(m, dict):
            msg = m.get("message") or m.get("msg") or ""
            details = m.get("details", {}) or {}

            rid = details.get("ruleId") or details.get("rule_id")
            ref = details.get("reference") or details.get("ref") or ""

            if rid:
                rule_ids.append(str(rid))
            if ref:
                refs.append(str(ref))
            if msg:
                messages.append(msg)
                m2 = TOTAL_SCORE_REGEX_JSON.search(msg)
                if m2:
                    total_score = int(m2.group(1))

        elif isinstance(m, str):
            messages.append(m)

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
        "messages": messages,
        "refs": refs,
        "total_score": total_score
    }


def get_latest_tx_info(audit_text: str):
    objs = parse_all_modsec_json_lines(audit_text)
    if not objs:
        return None
    return extract_tx_info(objs[-1])


def get_all_uids(audit_text: str):
    objs = parse_all_modsec_json_lines(audit_text)
    uids = []
    for obj in objs:
        info = extract_tx_info(obj)
        if info["uid"]:
            uids.append(info["uid"])
    return uids


# ---------------- 判斷是否被 WAF 擋下 ----------------
def is_blocked(http_code, rule_ids, total_score):
    if http_code in BLOCK_HTTP_CODES:
        return 1
    if len(rule_ids) > 0 and total_score > 0:
        return 1
    return 0


# ---------------- 發送 payload 並檢查 WAF ----------------
def post_and_check_modsecurity(payload: str):
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"message": payload}

    audit_before = read_docker_audit_log(DOCKER_CONTAINER, MODSEC_LOG_PATH)
    seen_uids = set(get_all_uids(audit_before))

    http_code = None
    response_text = ""

    try:
        resp = requests.post(
            POST_URL,
            headers=headers,
            data=data,
            timeout=REQUEST_TIMEOUT
        )
        http_code = resp.status_code
        response_text = resp.text[:500]
    except Exception as e:
        print(f"[POST ERROR] {e}")
        http_code = None

    matched_info = None

    for _ in range(MAX_WAIT_ROUNDS):
        t.sleep(SLEEP_AFTER_POST + LOG_WRITE_WAIT)
        audit_after = read_docker_audit_log(DOCKER_CONTAINER, MODSEC_LOG_PATH)
        objs = parse_all_modsec_json_lines(audit_after)

        if not objs:
            continue

        # 由後往前找最新且尚未出現過的 transaction
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

    return {
        "detected": detected,
        "rule_ids": rule_ids,
        "http_code": http_code,
        "response_text": response_text
    }


# ---------------- 計算指標 ----------------
def calc_metrics(df):
    labeled_df = df[df["label"].notna()].copy()

    if labeled_df.empty:
        return None

    y_true = labeled_df["label"].astype(int)
    y_pred = labeled_df["detected"].astype(int)

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
        "total_labeled": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall_tpr": recall,
        "f1": f1,
        "fpr": fpr,
        "tnr_specificity": tnr
    }


# ---------------- 輸出摘要 ----------------
def save_summary(df, metrics, out_txt):
    total = len(df)
    blocked = int((df["detected"] == 1).sum())
    bypassed = int((df["detected"] == 0).sum())

    lines = []
    lines.append("=== WAF Evaluation Summary ===")
    lines.append(f"Total samples: {total}")
    lines.append(f"Blocked: {blocked}")
    lines.append(f"Bypassed: {bypassed}")
    lines.append(f"Block rate: {blocked / total:.4f}" if total else "Block rate: 0.0000")
    lines.append(f"Bypass rate: {bypassed / total:.4f}" if total else "Bypass rate: 0.0000")
    lines.append("")

    if metrics is not None:
        lines.append("=== Classification Metrics (requires labels) ===")
        lines.append(f"Labeled samples: {metrics['total_labeled']}")
        lines.append(f"TP: {metrics['tp']}")
        lines.append(f"TN: {metrics['tn']}")
        lines.append(f"FP: {metrics['fp']}")
        lines.append(f"FN: {metrics['fn']}")
        lines.append(f"Accuracy: {metrics['accuracy']:.4f}")
        lines.append(f"Precision: {metrics['precision']:.4f}")
        lines.append(f"Recall / TPR: {metrics['recall_tpr']:.4f}")
        lines.append(f"F1-score: {metrics['f1']:.4f}")
        lines.append(f"FPR: {metrics['fpr']:.4f}")
        lines.append(f"Specificity / TNR: {metrics['tnr_specificity']:.4f}")
    else:
        lines.append("=== Classification Metrics ===")
        lines.append("No ground-truth labels found, so full accuracy/precision/recall cannot be computed.")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------- WAF 測試主流程 ----------------
async def stage_waf_only(filepath):
    print("\n=== [WAF Only] ModSecurity / CRS 測試 ===")

    if not check_container_running(DOCKER_CONTAINER):
        raise RuntimeError(
            f"找不到正在執行的容器: {DOCKER_CONTAINER}\n"
            f"請先執行 docker compose up -d"
        )

    if not wait_for_waf_ready():
        raise RuntimeError(
            f"WAF 服務尚未就緒: {POST_URL}\n"
            f"請確認 docker compose 已啟動，且 80 port 可連線"
        )

    if not audit_log_exists(DOCKER_CONTAINER, MODSEC_LOG_PATH):
        raise RuntimeError(
            f"找不到 audit log: {MODSEC_LOG_PATH}\n"
            f"請確認 docker-compose.yaml 已加入 MODSEC_AUDIT_LOG 設定，並重新 docker compose up -d"
        )

    print(f"[INFO] 使用 container: {DOCKER_CONTAINER}")
    print(f"[INFO] POST_URL: {POST_URL}")
    print(f"[INFO] MODSEC_LOG_PATH: {MODSEC_LOG_PATH}")

    data_df = load_payloads(filepath)
    rows = []

    for i, row in data_df.iterrows():
        payload = row["payload"]
        true_label = row["label"]

        result = await asyncio.to_thread(post_and_check_modsecurity, payload)
        pred = result["detected"]

        outcome = ""
        if true_label is not None:
            if int(true_label) == 1 and pred == 1:
                outcome = "TP"
            elif int(true_label) == 1 and pred == 0:
                outcome = "FN"
            elif int(true_label) == 0 and pred == 1:
                outcome = "FP"
            elif int(true_label) == 0 and pred == 0:
                outcome = "TN"

        rows.append({
            "index": i + 1,
            "payload": payload,
            "label": true_label,
            "http_code": result["http_code"],
            "rule_id": ",".join(result["rule_ids"]),
            "rule_count": len(result["rule_ids"]),
            "detected": pred,
            "outcome": outcome
        })

        print(
            f"[{i+1}/{len(data_df)}] "
            f"http={result['http_code']} "
            f"detected={pred} "
            f"rule_count={len(result['rule_ids'])} "
            f"label={true_label} "
            f"outcome={outcome}"
        )

    df = pd.DataFrame(rows)
    metrics = calc_metrics(df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("res/stats", exist_ok=True)

    out_csv = f"res/stats/waf_eval_results_{timestamp}.csv"
    out_txt = f"res/stats/waf_eval_summary_{timestamp}.txt"

    df[["index", "payload", "label", "http_code", "rule_id", "rule_count", "detected", "outcome"]].to_csv(
        out_csv, index=False, encoding="utf-8-sig"
    )

    save_summary(df, metrics, out_txt)

    print(f"\n✅ 結果已儲存到: {out_csv}")
    print(f"✅ 摘要已儲存到: {out_txt}")

    if metrics is not None:
        print("\n=== Metrics ===")
        print(f"TP       : {metrics['tp']}")
        print(f"TN       : {metrics['tn']}")
        print(f"FP       : {metrics['fp']}")
        print(f"FN       : {metrics['fn']}")
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall_tpr']:.4f}")
        print(f"F1-score : {metrics['f1']:.4f}")
        print(f"FPR      : {metrics['fpr']:.4f}")
    else:
        total = len(df)
        blocked = int((df["detected"] == 1).sum())
        bypassed = int((df["detected"] == 0).sum())
        print("\n=== No Labels Found ===")
        print(f"Block rate : {blocked / total:.4f}" if total else "Block rate : 0.0000")
        print(f"Bypass rate: {bypassed / total:.4f}" if total else "Bypass rate: 0.0000")

    return df


# ---------------- 主程式 ----------------
async def main():
    await stage_waf_only(INPUT_FILE)


if __name__ == "__main__":
    asyncio.run(main())