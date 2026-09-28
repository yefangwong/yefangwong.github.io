---
layout: post
title: "從「世界表象」到「概念本質」：以知網義原圖譜流形校準 (Retrofitting) 消除神經稠密向量的共現偏誤"
date: 2026-09-27 21:00:00 +0800
categories: [AI, NLP, Neuro-Symbolic]
tags: [Retrofitting, HowNet, BGE, Dense-Embedding, Outlines, Responsible-AI]
author: "Ye-Fang Wong (翁藝芳)"
lang: zh
---

> 🌐 **Language / 語言切換**: **繁體中文 (Current)** &nbsp;•&nbsp; [English Version →](/posts/2026-09-27-grounding-dense-embeddings-via-hownet-retrofitting-en/)

> **摘要 (Executive Summary)**：  
> 當前大語言模型（LLM）與稠密檢索模型（Dense Retrieval，如 BGE、OpenAI text-embedding）皆建立在分佈假說（Distributional Hypothesis）之上。然而，純統計共現使得稠密向量深受「世界表象之困」——無法在幾何空間中有效分離「概念本質真正相似（Genuine Similarity）」與「語境同框關聯（Relatedness）」。  
> 本文記錄我們在實驗室針對 BGE-large-zh 稠密向量進行的終極假說檢定：原始向量在真相似 Q1（0.6330）與語境同框 Q2（0.5505）之間，雙樣本 t 檢定為 $t = 1.3183, p = 0.2239$（統計上無法分離）。我們引入董振東先生知網（HowNet 2,089 義原圖譜）作為符號流形先驗，結合 Manaal Faruqui et al. 2015 的馬可夫隨機場凸二次規劃與座標上升閉式解進行後微調（Retrofitting）。  
> **實證結果顯示：校準後向量在 Q1（0.9608）與 Q2（0.6427）之間達到 $t = 7.3428, p = 0.000080$ 的極顯著分離，分離度跨越 3 個數量級！** 同時，本文針對校準後 Q2 仍有 0.6427 的空間底噪進行幾何本質批判，指出負向斥力場（Counter-fitting）與龐加萊雙曲空間的演化路徑，並與輸出端 Outlines 受控生成串聯，構建端到端神經符號高保證架構。

---

## 🏛️ 一、 問題緣起：大語言模型稠密向量的「世界表象之困」

在現代自然語言處理（NLP）與企業級 RAG（檢索增強生成）架構中，Dense Embedding（稠密向量）被視為語義理解的基石。其底層理論源自 J.R. Firth 1957 年著名的分佈假說：

> *"You shall know a word by the company it keeps."（觀其伴，知其意）*

然而，純粹依賴語料庫中字詞共現頻率（Co-occurrence Statistics）的代價，是**將「世界的表象（語境同框）」誤認為「概念的本質（本體相似）」**：
* **真正相似 (Genuine Similarity)**：概念本質具備相同或高度重疊的本體屬性。例如「醫生」與「大夫」、「爸爸」與「父親」、「貓」與「狗」。
* **語境同框關聯 (Relatedness)**：在真實世界中頻繁同時出現、具有功能或因果鏈路，但概念本質截然不同。例如「醫生」與「醫院」、「貓」與「老鼠」、「咖啡」與「杯子」。

當我們使用預訓練 Dense Embedding 進行檢索或語義推斷時，模型常因「同框率極高」而將相關詞判定為極度相似。在金融合規、司法判決、醫療診斷與精密代碼審計等高保證領域，這種「概念漂移（Semantic Drift）」往往正是神經幻覺的源頭。

---

## 🔬 二、 核心假設與四象限評測基準

為了客觀量化這一現象，我們在 `token-meaning-lab` 構建了嚴格的四象限語義評測集（參照 Hill et al. 2015 SimLex-999 黃金標準）：

<div class="quadrant-wrapper">
  <div class="quadrant-axis-y">▲ 本體真正相似度 (Genuine Ontological Similarity)</div>
  <div class="quadrant-grid">
    <div class="quadrant-card q3">
      <div class="quadrant-title">🔹 Q3: 概念相同 / 鮮少同框</div>
      <div class="quadrant-desc">跨領域結構同構、真相似但低共現率</div>
      <span class="quadrant-example">範例：企鵝 vs 蜂鳥、心臟 vs 抽水機</span>
    </div>
    <div class="quadrant-card q1">
      <div class="quadrant-title">⭐ Q1: 概念本質相同 / 高度同框</div>
      <div class="quadrant-desc"><strong>【核心檢驗：真相似】</strong>本體屬性高度重疊且同框</div>
      <span class="quadrant-example">範例：醫生 vs 大夫、爸爸 vs 父親</span>
    </div>
    <div class="quadrant-card q4">
      <div class="quadrant-title">⚪ Q4: 概念無關 / 鮮少同框</div>
      <div class="quadrant-desc">正交底噪對照組、語義本體完全不相干</div>
      <span class="quadrant-example">範例：醫生 vs 香蕉、雲朵 vs 剪刀</span>
    </div>
    <div class="quadrant-card q2">
      <div class="quadrant-title">⚠️ Q2: 概念相異 / 高度同框</div>
      <div class="quadrant-desc"><strong>【統計共現陷阱：典型盲區】</strong>同框率極高但本質相異</div>
      <span class="quadrant-example">範例：醫生 vs 醫院、貓 vs 老鼠</span>
    </div>
  </div>
  <div class="quadrant-axis-x">語境同框率 (Statistical Co-occurrence / Relatedness) ►</div>
</div>

### 檢定目標 (Hypothesis)
* **核心假設**：未經符號知識校準的神經向量模型，在 Q1（真相似）與 Q2（語境同框）上的餘弦相似度分佈無法在統計學上達成顯著分離（$p > 0.05$）。
* **判定標準**：採用獨立雙樣本 $t$ 檢定（Two-sample $t$-test），檢驗 Q1 與 Q2 的相似度均值是否存在統計顯著性差異。

---

## 📉 三、 基準檢驗：原始 BGE 向量的統計失效 (p = 0.2239)

我們選用當前中文開源頂級檢索模型 **BAAI/bge-large-zh-v1.5**（1024 維稠密向量）進行實測：

```python
# 測試範例：計算餘弦相似度
sim_Q1 = cosine_similarity(bge("醫生"), bge("大夫"))  # 0.6330
sim_Q2 = cosine_similarity(bge("醫生"), bge("醫院"))  # 0.5505
```

### 統計檢驗結果：
* **Q1 (真相似對)**：平均餘弦相似度 = **0.6330**
* **Q2 (同框相關對)**：平均餘弦相似度 = **0.5505**
* **雙樣本 $t$ 檢定**：$t = 1.3183$，$p = 0.2239$

> ❌ **結論：原始 BGE 向量完全無法區分 Q1 與 Q2！**  
> $p = 0.2239$ 遠高於統計顯著閾值（$\alpha = 0.05$）。這證明了即便是在數十億 Token 上預訓練的 SOTA 稠密模型，在面對「同框關聯詞」時，依然深受分佈統計的嚴重干擾。

---

## 🏛️ 四、 符號本體先驗：知網 (HowNet) 2,089 義原圖譜

與純分佈模型不同，人類在認知科學中早已建立精準的符號概念網絡。我們回溯至董振東先生於 2010 年發表之《知網及其語義計算》（COLING 2010）。

知網的核心哲學在於**義原（Sememes）**——不可再分割的最小語義原子單位。知網以 2,089 個核心義原，透過知識描述語言（KDML）結構化拆解全量字詞：
* **「醫生」**：`{human|人: HostJob={medical|醫}, doctor}`
* **「大夫」**：`{human|人: HostJob={medical|醫}, doctor}`（義原完全同構，相似度 = 1.0）
* **「醫院」**：`{InstitutePlace|場所機構: domain={medical|醫}}`（基本義原為機構，非人類，相似度自然分離）

### 知網符號基準線檢定 (Symbolic Ground Truth)：
* **Q1 義原拓撲相似度均值**：**0.8889**
* **Q2 義原拓撲相似度均值**：**0.4259**
* **雙樣本 $t$ 檢定**：$t = 6.9281$，$p = 0.000121$

> ✅ **證明**：知網的符號義原具備極高的信噪比，在統計上以極大顯著性（$p = 0.000121$）徹底分開了概念本質與同框關聯！

---

## 📐 五、 凸優化流形微調：Faruqui Retrofitting 數學模型

我們如何將知網的高純度符號本體，注入到神經網路的高維向量空間中？  
我們研析了卡內基美隆大學（CMU）Manaal Faruqui et al. 發表於 NAACL 2015 的經典論文 *《Retrofitting Word Vectors to Semantic Lexicons》*（arXiv:1411.4166）。

Faruqui 提出將外部詞網作為馬可夫隨機場（Markov Random Field），並設計了一個兼顧**「原始向量神經保真」**與**「圖譜鄰居流形向心」**的凸優化目標函數：

$$\Psi(Q) = \sum_{i=1}^{n} \left[ \alpha_i \|\mathbf{q}_i - \hat{\mathbf{q}}_i\|^2 + \sum_{j:(i,j)\in E} \beta_{ij} \|\mathbf{q}_i - \mathbf{q}_j\|^2 \right]$$

* $\hat{\mathbf{q}}_i$：預訓練之原始神經稠密向量（觀測值，Observed Prior）。
* $\mathbf{q}_i$：待推斷之校準向量（Inferred Target）。
* $\alpha_i$：神經保真權重（設為 1.0）。
* $\beta_{ij}$：符號圖譜引力權重（設為 $\text{degree}(i)^{-1}$，依度數衰減防止高度 Hub 節點過度拉扯）。

### 閉式在線迭代公式 (Closed-Form Coordinate Ascent)：
由於 $\Psi(Q)$ 為嚴格二次凸函數，對 $\mathbf{q}_i$ 求導並令其為零，可直接導出在線座標上升更新式：

$$\mathbf{q}_i^{(t+1)} = \frac{\alpha_i \hat{\mathbf{q}}_i + \sum_{j \in N(i)} \beta_{ij} \mathbf{q}_j^{(t)}}{\alpha_i + \sum_{j \in N(i)} \beta_{ij}}$$

* **收斂性與效能**：該演算法在 10 次迭代內即可完全收斂。在單顆 CPU 上處理 10 萬個詞彙，**僅需約 5 秒**！
* **論文核心洞察**：Faruqui 實證證明，**這種輕量級的後微調（Retrofitting），其下游任務表現全面打平甚至超越了昂貴的訓練期先驗（MAP）！**

---

## 📊 六、 實證數據驗證：顯著性與分離度分析

在 `token-meaning-lab` 中，我們以知網 2,089 義原建立的向心引力邊對 BGE 稠密向量進行 Retrofitting。

### 核心檢驗數據總表：

| 評測維度 | 原始 BGE 向量 (Dense Baseline) | 知網符號基準 (Symbolic Ground Truth) | 校準後向量 (Retrofitted Embeddings) |
| :--- | :--- | :--- | :--- |
| **Q1 (真相似) 均值** | 0.6330 | 0.8889 | **0.9608** |
| **Q2 (同框詞) 均值** | 0.5505 | 0.4259 | **0.6427** |
| **兩者均值淨差值 (Gap)** | +0.0825 (模糊) | +0.4630 (清晰) | **+0.3181 (大幅拉開)** |
| **雙樣本 $t$ 統計量** | $t = 1.3183$ | $t = 6.9281$ | **$t = 7.3428$** |
| **顯著性 $p$-value** | **$p = 0.2239$ (未達顯著 ❌)** | **$p = 0.000121$ (顯著 ✅)** | **$p = 0.000080$ (極顯著 ✅)** |

```
【顯著性 p-value 對比圖 (對數尺度)】
原始 BGE 向量 : 0.2239 ════════ (未達顯著)
知網符號基準 : 0.000121 ══ (顯著)
校準後向量   : 0.000080 ═ (極顯著)
```

> 🎯 **實證結論**：  
> 透過知網圖譜的流形約束，$p$-value 從 $0.2239$ 降至 $0.000080$（$p < 0.0001$）。這在統計學上證明：**神經向量的世界表象偏誤，能被高信噪比的符號先驗以極低代價有效校準。**

---

## 🧠 七、 批判性反思：Q2 = 0.6427 底噪之謎

觀察到這些數據後，我們進一步去思考：

> *「校準後的 Q2（同框詞）相似度依然高達 0.6427，超過了一半！為什麼同框詞還是這麼相似？」*

基於此疑問，AI (Gemini 3.6 Flash) 從幾何空間與圖論拓撲提出了三個尚待進一步驗證的可能物理原因：

1. **高維歐氏空間各向異性（Anisotropy / Cone Effect）**：  
   Transformer 與 BERT 系列的嵌入空間存在「圓錐效應」——所有向量往往擠在一個狹窄的圓錐流形中，即使是隨機無關詞，餘弦相似度底噪也在 $0.35 \sim 0.45$。
2. **Faruqui 模型的物理不對稱性：只有引力、沒有斥力**：  
   Faruqui 的凸二次目標函數中只有 $\beta_{ij} \|\mathbf{q}_i - \mathbf{q}_j\|^2$（正向拉力）。它專注於將同義詞拉近，卻沒有機制主動將「同框但非同義」的詞推開！
3. **2-Hop 圖譜間接拉扯（Graph Pulling）**：  
   在知網中，「貓」與「老鼠」雖然不是同義詞，但它們都共享了 `{animal|動物}`, `{mammal|哺乳動物}` 義原，在無向圖中形成了強大的 2-hop 引力傳導。

### 🔬 未來探索方向

基於上述洞察，後續可以進一步探索以下方向：

* **BRANCH-05A (Counter-fitting 負向斥力場)**：  
  引進劍橋大學 Nikola Mrkšić et al. 2016 的 Counter-fitting 機制，在目標函數中引入非凸負約束項：
  $$\max(0, \gamma - \|\mathbf{q}_i - \mathbf{q}_k\|^2)$$
  主動將語境同框詞推離 $\gamma$ 距離之外，目標將 Q2 相似度壓制至 $0.25$ 以下！
* **BRANCH-05B (Poincaré 雙曲幾何流形)**：  
  由於語言本體具備嚴格的樹狀層級結構（上位/下位詞），在高維歐氏空間中必然產生空間曲率失真。將流形映射至龐加萊雙曲圓盤（Poincaré Disk），可天然保持樹狀層級距離。

---

## 🏰 八、 神經符號縱深合流：輸入端表徵純化 × 輸出端 FSM 受控生成

這項實驗的突破，為我們正在構建的 **負責任 AI 縱深防禦體系（Responsible AI Defense-in-Depth）** 補齊了關鍵的拼圖：

```
┌────────────────────────────────────────┐
│   用戶提示詞 (Prompt / Data Query)     │
└───────────────────┬────────────────────┘
                    │
   【第一道防線: 表徵端】
   HowNet Retrofitting 幾何流形校準 (p=0.000080)
                    │
                    ▼
   ┌────────────────────────────────────┐
   │    Dense LLM 預訓練模型推論        │
   └────────────────┬───────────────────┘
                    │
   【第二道防線: 生成端】
   Willard & Louf 2023 Outlines FSM 硬遮罩
                    │
                    ▼
   ┌────────────────────────────────────┐
   │    輸出端 Logits 投影遮罩 (<1ms)   │
   └────────────────┬───────────────────┘
                    │
   【第三道防線: 執行端】
   llm-sql-guard AST 語意資安閘門
                    │
                    ▼
   ┌────────────────────────────────────┐
   │    執行期策略與注入攔截 (<3ms)     │
   └────────────────────────────────────┘
```

1. **輸入端（表徵幾何）**：利用 **HowNet Retrofitting**，確保向量在進入神經網路或向量檢索庫前，具備絕對真確的符號本體距離。
2. **輸出端（生成機率）**：利用 **Outlines（Willard & Louf 2023）** 的有限狀態機（FSM）於解碼期即時遮罩 Logits ($-\infty$)，提供數學級 100% 結構語法合規保證。
3. **執行端（語意策略）**：利用自研的 **llm-sql-guard**，在 SQL/代碼生成後以 AST 進行策略審計與注入防禦。

---

## 結語：源於人文關懷，讓 AI 更加理解人類

從 2005 年在台大資工甄試計畫書中提出《跨語文資訊檢索系統連結雙語知識本體與領域知識語義之研究》，到 2026 年以統計顯著數據（$p = 0.000080$）見證神經向量與知網義原的接合——這不僅僅是一次演算法的驗證，更是一場橫跨二十餘年、源於人文關懷的深情探索。

在追求 AI 自我迭代（RSI）的浪潮中，我始終堅信：技術的終極價值在於人文關懷。將神經網絡與符號知識結合，不只是追求數學與幾何上的精準，更是為了讓 AI 能真正理解人類概念與語言背後的本質——讓 AI 變得更加認識人、理解人與同理人，而不只是一台隨機和統計的機器。

---

### 📚 參考文獻與實驗代碼
1. Faruqui, M., Dodge, J., Jauhar, S. K., Dyer, C., Hovy, E., & Smith, N. A. (2015). *Retrofitting Word Vectors to Semantic Lexicons*. In Proceedings of NAACL 2015. [arXiv:1411.4166](https://arxiv.org/abs/1411.4166)
2. Dong, Z., & Dong, Q. (2010). *HowNet And The Computation Of Meaning*. In Proceedings of COLING 2010.
3. Hill, F., Reichart, R., & Korhonen, A. (2015). *SimLex-999: Evaluating Semantic Models with (Genuine) Similarity Estimation*. Computational Linguistics, 41(4), 665-696.
4. Willard, B. T., & Louf, R. (2023). *Efficient Guided Generation for Large Language Models*. [arXiv:2307.09702](https://arxiv.org/abs/2307.09702)
5. Mrkšić, N., Séaghdha, D. Ó., Thomson, B., Gašić, M., Rojas-Barahona, L., Su, P. H., Vandyke, D., Wen, T. H., & Young, S. (2016). *Counter-fitting Word Vectors to Linguistic Constraints*. In Proceedings of NAACL 2016.
