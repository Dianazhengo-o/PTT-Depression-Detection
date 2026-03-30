# PTT-Depression-Detection
A dual-stream fusion network combining BERT and linguistic features for depression detection on PTT.

---

# Multi-Modal Depression Detection in Social Media (PTT)

本專案旨在利用自然語言處理（NLP）與深度學習技術，針對社群平台（PTT）文本進行憂鬱傾向檢測。透過整合手工語言學特徵（Handcrafted Features）與深度語意表徵（MentalBERT），構建高精準度的分類模型。

## 專案核心架構

本研究採用的偵測架構分為四個主要階段：

1.  **文本預處理**：
    * 利用 `OpenCC` 進行繁簡轉換。
    * 使用 `CKIP Transformers` 執行精確的中文斷詞（WS）與詞性標註（POS）。
2.  **特徵工程 (Feature Engineering)**：
    * **深度語意特徵**：提取 `Chinese-MentalBERT` 的 [CLS] 向量（768 維）。
    * **語言學特徵**：基於心理學文獻提取第一人稱單數（I-talk）、絕對主義詞（Absolutist words）、認知機制詞、否定詞及詞性分布。
    * **行為特徵**：分析發文時間節律（如凌晨 0-6 點之高風險時段）。
3.  **模型設計**：
    * **MLP Fusion Net**：採用雙分支神經網路，分別處理高維語意向量與數值型語言學特徵，最終透過拼接（Concatenation）進行特徵融合與分類。
4.  **統計檢定**：
    * 利用 Mann-Whitney U 檢定與 Cohen's d 評估各項特徵在憂鬱組與對照組間的顯著性差異。

## 實驗結果

在 `Prozac`（憂鬱組）與 `Gossiping`（對照組）的平衡資料集上，**MLP Fusion 模型** 的 5-Fold 交叉驗證表現如下：

| 指標 | 數值 |
| :--- | :--- |
| **平均準確率 (Accuracy)** | **97.26% (± 0.53%)** |
| **平均 F1-Score** | **0.9731** |
| **Precision (Prozac)** | **0.9868** |
| **Recall (Prozac)** | **0.9574** |

## 技術棧

* **Language**: Python 3.10+
* **NLP Tools**: CKIP Transformers, OpenCC
* **Deep Learning**: PyTorch, HuggingFace Transformers
* **Models**: zwzzz/Chinese-MentalBERT
* **Statistical Analysis**: Pandas, Scipy, Scikit-learn

---

## 原始程式碼與執行環境

由於 GitHub 預覽限制，如需查看完整的互動式代碼與執行紀錄，請至 Google Colab 存取原始檔案。

**Google Colab Notebook Link:**
```text
https://colab.research.google.com/drive/1oeSXA2O_t-xSp8dpphWUdIPPfAP_VO5M?usp=sharing
```

---
