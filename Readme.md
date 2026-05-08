# WAF_ML

本專案是一個結合 **Web Application Firewall (WAF)** 與 **Machine Learning (ML)** 的分析系統，  
用於研究與評估 XSS payload 在 WAF 中的偵測行為，並進一步透過機器學習模型進行分類與預測。

---

# 專案在做什麼（核心概念）

傳統 WAF（如 OWASP CRS）依賴人工維護規則，但現代攻擊 payload 會透過：

- Encoding（Base64、HTML Entity、UTF-16）
- Obfuscation（關鍵字拆分、混淆）
- Payload mutation（自動變異）
- 語法重組

導致新型攻擊可能在規則更新前繞過 WAF。

---

## 本專案的核心方法

本系統將 WAF 的「偵測過程」轉換成機器學習可以學習的資料：

---

## 系統目標

- 分析 WAF 規則（CRS）觸發行為
- 建立 payload 的機器學習分類模型
- 預測 payload 是否為惡意
- 預測 payload 是否會被 WAF 阻擋
- 分析哪些 CRS rule 最重要
- 支援未來自動化攻防系統（LLM / DDQN）

---

# 系統流程（完整 Pipeline）
## 一、模型訓練流程（第一次一定要跑）

Step 1：資料切分
python src/stage1.py

Step 2：建立 Rule 字典
python src/stage2.py

Step 3：特徵轉換
python src/stage3.py

Step 4：模型訓練
python src/stage4.py

Step 4-2：模型評估（報告用）
python src/stage4_2.py

Step 5：Rule 權重分析
python src/stage5.py

###WAF + ML 預測（實際測試）
python src/ml_prediction.py