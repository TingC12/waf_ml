#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import joblib
import requests
import pandas as pd
import subprocess
from typing import List, Optional

# =========================
# Config
# =========================
INPUT_FILE = "res/llm_output/success_payloads_0.7_01152305.txt"   # .txt 或 .csv 都可
OUTPUT_CSV = "res/predict/waf_ml_pipeline_results.csv"

MODEL_PATH = "models/log_reg_rules_l2.joblib"
RULE_VOCAB_JSON = "res/features/rule_vocab.json"

DOCKER_CONTAINER = "waf"
MODSEC_LOG_PATH = "/tmp/modsec_audit.log"
POST_URL = "http://localhost:80/comment"

REQUEST_TIMEOUT = 5
SLEEP_AFTER_POST = 0.15
LOG_WRITE_WAIT = 0.05
MAX_WAIT_ROUNDS = 15

BLOCK_HTTP_CODES = {403, 406, 429, 500, 501, 502, 503}
TOTAL_SCORE_REGEX_JSON = re.compile(r"Total\s*Score:\s*(\d+)", re.IGNORECASE)

CSV_PAYLOAD_COLUMNS = ["payload", "Payload", "text", "input"]

# =========================
# Utils
# =========================
def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def run_cmd(cmd: List[str], check: bool = True) -> str:
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result.stdout.strip()


def docker_exec(container: str, shell_cmd: str, check: bool = True) -> str:
    return run_cmd(["docker", "exec", container, "sh", "-lc", shell_cmd], check=check)


def read_modsec_log() -> str:
    return docker_exec(DOCKER_CONTAINER, f"cat {MODSEC_LOG_PATH} 2>/dev/null || true", check=False)


def load_rule_vocab(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    if not isinstance(vocab, list):
        raise ValueError("rule_vocab.json 格式錯誤，應該要是 list")
    return [str(x).strip() for x in vocab if str(x).strip()]


def load_model(path: str):
    return joblib.load(path)


def get_model_score(model, X: pd.DataFrame) -> Optional[float]:
    if hasattr(model, "decision_function"):
        val = model.decision_function(X)
        if hasattr(val, "__len__"):
            return float(val[0])
        return float(val)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)
        return float(prob[0, 1])
    return None


def find_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def read_csv_with_fallback(input_file: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp1252", "big5", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            print(f"[INFO] trying encoding: {enc}")
            return pd.read_csv(input_file, encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e
            continue

    raise ValueError(f"無法讀取 CSV，請確認檔案編碼。最後錯誤: {last_error}")


def load_payloads(input_file: str) -> pd.DataFrame:
    ext = os.path.splitext(input_file)[1].lower()

    if ext == ".txt":
        rows = []
        encodings = ["utf-8-sig", "utf-8", "cp1252", "big5", "latin1"]

        for enc in encodings:
            try:
                with open(input_file, "r", encoding=enc, errors="strict") as f:
                    for idx, line in enumerate(f):
                        payload = line.rstrip("\n")
                        if payload.strip() == "":
                            continue
                        rows.append({
                            "index": idx,
                            "payload": payload
                        })
                print(f"[INFO] txt encoding: {enc}")
                return pd.DataFrame(rows)
            except UnicodeDecodeError:
                rows = []
                continue

        raise ValueError("TXT 檔案無法解碼，請確認編碼格式")

    elif ext == ".csv":
        df = read_csv_with_fallback(input_file)
        payload_col = find_first_existing_column(df, CSV_PAYLOAD_COLUMNS)

        if payload_col is None:
            raise ValueError(f"{input_file} 找不到 payload 欄位，支援欄位：{CSV_PAYLOAD_COLUMNS}")

        out = pd.DataFrame()
        out["index"] = range(len(df))
        out["payload"] = df[payload_col].fillna("").astype(str)
        return out

    else:
        raise ValueError("INPUT_FILE 只支援 .txt 或 .csv")


# =========================
# 第一份檔案的 audit log 抓法
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
        "rule_ids": list(dict.fromkeys(rule_ids)),
        "messages": messages,
        "refs": refs,
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
# Feature engineering
# =========================
def build_feature_row(payload: str, rule_ids: List[str], vocab: List[str]) -> pd.DataFrame:
    fired = set(str(x).strip() for x in rule_ids if str(x).strip())

    row = {
        "payload": payload,
        "rule_count": len(fired),
    }

    for rid in vocab:
        row[f"rule_{rid}"] = 1 if rid in fired else 0

    return pd.DataFrame([row])


def align_feature_columns_for_model(X_df: pd.DataFrame, model, vocab: List[str]) -> pd.DataFrame:
    expected = [f"rule_{rid}" for rid in vocab]
    if "rule_count" in X_df.columns:
        expected = ["rule_count"] + expected

    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)

    for col in expected:
        if col not in X_df.columns:
            X_df[col] = 0

    X_df = X_df[expected].fillna(0)
    return X_df


# =========================
# WAF request
# =========================
def send_payload(payload: str):
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }
    data = {"message": payload}

    try:
        response = requests.post(
            POST_URL,
            headers=headers,
            data=data,
            timeout=REQUEST_TIMEOUT
        )
        return response.status_code, response.text[:500]
    except requests.RequestException as e:
        return None, str(e)


def is_blocked(http_code, rule_ids, total_score):
    if http_code in BLOCK_HTTP_CODES:
        return 1
    if len(rule_ids) > 0 and total_score > 0:
        return 1
    return 0


def get_rules_for_payload(payload: str):
    audit_before = read_modsec_log()
    seen_uids = set(get_all_uids(audit_before))

    http_code, response_text = send_payload(payload)

    matched_info = None

    for _ in range(MAX_WAIT_ROUNDS):
        time.sleep(SLEEP_AFTER_POST + LOG_WRITE_WAIT)
        audit_after = read_modsec_log()
        objs = parse_all_modsec_json_lines(audit_after)

        if not objs:
            continue

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

    waf_blocked = is_blocked(http_code, rule_ids, total_score)

    return http_code, response_text, waf_blocked, rule_ids, len(rule_ids), total_score


# =========================
# Summary
# =========================
def summarize(df: pd.DataFrame, pred_col: str, title: str):
    print(f"\n=== {title} ===")
    total = len(df)
    pos = int((df[pred_col] == 1).sum())
    neg = int((df[pred_col] == 0).sum())
    print(f"Total: {total}")
    print(f"Positive(1): {pos}")
    print(f"Negative(0): {neg}")


def main():
    ensure_parent_dir(OUTPUT_CSV)

    print("[INFO] loading model...")
    model = load_model(MODEL_PATH)

    print("[INFO] loading vocab...")
    vocab = load_rule_vocab(RULE_VOCAB_JSON)

    print("[INFO] loading payloads...")
    df_input = load_payloads(INPUT_FILE)

    print(f"[INFO] total payloads: {len(df_input)}")
    print(f"[INFO] vocab size: {len(vocab)}")

    rows = []

    for i, row in df_input.iterrows():
        payload = str(row["payload"])

        http_code, response_text, waf_blocked, rule_ids, rule_count, total_score = get_rules_for_payload(payload)

        feat_df = build_feature_row(payload, rule_ids, vocab)
        X = align_feature_columns_for_model(
            feat_df.drop(columns=["payload"], errors="ignore"),
            model,
            vocab
        )

        ml_pred = int(model.predict(X)[0])
        ml_score = get_model_score(model, X)

        rows.append({
            "index": int(row["index"]),
            "payload": payload,
            "http_code": http_code,
            "waf_blocked": waf_blocked,
            "rule_id": ",".join(rule_ids),
            "rule_count": rule_count,
            "total_score": total_score,
            "ml_pred": ml_pred,
            "ml_score": ml_score,
            "waf_ml_same": int(waf_blocked == ml_pred),
            "response_preview": response_text
        })

        print(
            f"[{i+1}/{len(df_input)}] "
            f"http={http_code} waf={waf_blocked} rules={rule_count} ml={ml_pred}"
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n[DONE] saved: {OUTPUT_CSV}")

    summarize(out_df, "waf_blocked", "WAF Summary")
    summarize(out_df, "ml_pred", "ML Summary")

    diff_count = int((out_df["waf_blocked"] != out_df["ml_pred"]).sum())
    print(f"\n=== WAF vs ML Difference ===")
    print(f"Different predictions: {diff_count}/{len(out_df)}")


if __name__ == "__main__":
    main()