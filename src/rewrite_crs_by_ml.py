import joblib
import re

rule_ids = [
    "921130", "932100", "932110", "932115", "932130", "932150", "932160",
    "933100", "933130", "933160", "933210", "934100",
    "941100", "941110", "941120", "941130", "941140", "941160", "941170",
    "941180", "941190", "941210", "941220", "941230", "941250", "941260",
    "941270", "941290", "941300", "941310", "941370",
    "942100", "942230", "942360", "943100", "949110"
]

model = joblib.load("models/log_reg_rules_l2.joblib")

weights = model.coef_[0]

scale = 10
w_scaled = [round(w * scale) for w in weights]
rule_weight_map = dict(zip(rule_ids, w_scaled))

input_file = "REQUEST-941-APPLICATION-ATTACK-XSS.conf"
output_file = "REQUEST-941-APPLICATION-ATTACK-XSS-ML.conf"

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

blocks = re.split(r'(?=SecRule|SecAction)', content, flags=re.MULTILINE)
new_blocks = []

for block in blocks:
    match = re.search(r'\bid\s*:\s*[\'"]?(\d+)', block)
    if not match:
        new_blocks.append(block)
        continue

    rid = match.group(1)

    if rid not in rule_weight_map:
        new_blocks.append(block)
        continue

    w = rule_weight_map[rid]

    block = re.sub(r",?\s*setvar:\s*'tx\.xss_score=\+%?\{?tx\.[^']+\}?'", "", block)
    block = re.sub(r",?\s*setvar:\s*'tx\.inbound_anomaly_score_pl\d=\+%?\{?tx\.[^']+\}?'", "", block)
    block = re.sub(r",?\s*setvar:\s*tx\.xss_score=\+%?\{?tx\.[^,\s\"]+\}?", "", block)
    block = re.sub(r",?\s*setvar:\s*tx\.inbound_anomaly_score_pl\d=\+%?\{?tx\.[^,\s\"]+\}?", "", block)

    block = re.sub(r",\s*,", ",", block)
    block = re.sub(r'"\s*,', '"', block)

    # 只保留正權重
    if w > 0:
        pl_match = re.search(r'paranoia-level/(\d)', block)
        pl = pl_match.group(1) if pl_match else "1"

        new_actions = (
            f",setvar:'tx.xss_score=+{w}'"
            f",setvar:'tx.inbound_anomaly_score_pl{pl}=+{w}'"
        )
        block = re.sub(r'"\s*$', new_actions + '"', block)

    new_blocks.append(block)

with open(output_file, "w", encoding="utf-8") as f:
    f.write("".join(new_blocks))

print(f"✅ 已輸出: {output_file}")