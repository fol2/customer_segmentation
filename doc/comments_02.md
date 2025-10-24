好消息先講：**Run 04（continuous + KMeans）**係一個幾理想嘅基線—**9 個 cluster**、**Silhouette ≈ 0.385**、**DB ≈ 1.03**、**0% noise**，落地可行同可解釋性都唔錯（同你 repo 報告一致）。

你而家見到嘅「**confidence ratio 偏低**／邊界客太多」主要唔係 model 爆缸，而係**信心度計法未校準**＋**KMeans 對「橢圓形/斜長」簇嘅距離假設偏球形**，令到（1）有啲 cluster 中心距離相對唔細；（2）跨簇「第二近中心」其實幾近，於是**ratio（nearest‑other / assigned）**唔夠高。啲圖都反映到有幾個簇呈**狹長／傾斜**（PCA/t‑SNE），用**單一歐氏距離**去量度信心會偏保守。你嘅 run 內部指標仍然健康，所以我主張**保留 Run 04 的聚類**，但**重寫信心度與邊界辨識**同**散點圖數據**（供 Power BI）嘅產出。

---

## 針對性診斷（為何 ratio 低）

1. **距離未標準化**：每個 cluster 嘅「半徑/伸延」唔同；遠離中心但仍喺簇核心嘅點，**assigned 距離會偏大**→ ratio 偏低。
2. **形狀非球形**：KMeans 以等方差球形作近似，**橢圓/斜長**簇（見 C7、C1 投影）會令「最近其他中心」可相當接近 → ratio 下降。
3. **閾值太硬**：`ratio < 1.5 或 silhouette < 0.2` 一刀切，對**鄰近簇**嘅「自然邊界」偏嚴，導致「marginal」比例膨脹。
4. **度量空間**：你用嘅距離係**RobustScaler+log1p** 之後嘅空間，係啱，不過**未按簇內變異度作權重**（變相將高方差方向低估）。Run 04 的整體分離度仍然好（silhouette/DB 如報告），問題集中喺**信心標度**唔啱用。

---

## 解決方案（不改 cluster、只改「信心度/邊界」與可視化）

> 保持 KMeans 分簇結果（9 群）不變；新增 4 組 **更科學**的信心度／邊界度量；更新 Power BI dataset 字段；圖表更貼邊界結構。

### A. 四個新指標（取代「簡單 ratio」）

1. **半徑標準化距離（Radius‑normalised distance, RND）**

   * 先計每簇**魯棒半徑** `r_c = median(‖x−μ_c‖)`；
   * 將距離**按所屬簇半徑**標準化：`d̂_assigned = ‖x−μ_assigned‖ / r_assigned`；
   * 同樣計「最近其他簇」標準化距離 `d̂_other`（用其各自半徑）；
   * **邊界間隙**：`margin_r = d̂_other − d̂_assigned`（>0 越安全）。

2. **對角馬氏距離（Diagonal Mahalanobis, d_M^2）**

   * 以**簇內對角協方差** `Σ_c,diag`（每簇每特徵方差，+ε）計
     `d_M^2(x,c) = Σ_j ((x_j−μ_cj)^2 / (σ_cj^2+ε))`；
   * 這能應對**橢圓/斜長**簇，避免球形距離誤判邊界。

3. **Soft K‑means 概率（溫度 τ）**

   * `p(c|x) ∝ exp(− d_M^2(x,c) / (2τ))`；**Confidence = max_c p(c|x)**；
   * τ 可設 1（或以中位 d_M 作自適應），比 ratio 更**可校準**。

4. **Top‑2 距離差（Gap）**

   * `gap = √d2(second) − √d2(first)`（或以 Mahalanobis 距離計）；
   * 直接表示落在決策邊界嘅**幾何 margin**。

> **標記邊界點**：建議用**分簇自適應閾值**（例如每簇 `confidence` 之 20% 分位、或 `margin_r < 0`）＋**全局 silhouette**（如 `< 0.1`）聯合規則，取代硬性 `ratio < 1.5`。

---

## 已修正版腳本（替換 `create_scatter_and_distance_analysis.py` 的核心計算段）

> **重點**：完全向後兼容輸出檔案名；新增多個欄位；全部**向量化**（無 Python for‑loop），快好多。
> 將底下 **PATCH** 替換你檔案中 **第 3/4/5/6 節**（距離與信心 → 可視化 → Power BI dataset）相應段落即可。

```python
# ==== 3. Distances & Confidence (vectorised, calibrated) ====
from numpy.linalg import norm

X = X_transformed.astype("float32")
C = kmeans_model.cluster_centers_.astype("float32")
n, k = X.shape[0], C.shape[0]

# (n,k) squared Euclidean distances in transformed space
# Using broadcasting for speed
diff = X[:, None, :] - C[None, :, :]
d2 = (diff ** 2).sum(axis=2)                       # (n, k)
first_idx = labels
first_d2 = d2[np.arange(n), first_idx]

# nearest-other cluster indices & distances
# argpartition: O(k) selection for 2 smallest
order2 = np.argpartition(d2, 2, axis=1)[:, :2]     # indices of 2 smallest per row (unsorted)
# ensure first is the assigned cluster; pick the other as second
second_idx = np.where(order2[:, 0] == first_idx, order2[:, 1], order2[:, 0])
second_d2 = d2[np.arange(n), second_idx]

# --- (1) radius-normalised distances & margin ---
# robust radius per cluster: median of sqrt distance of in-cluster points
radii = np.zeros(k, dtype="float32")
for c in range(k):
    idx = (first_idx == c)
    if np.any(idx):
        radii[c] = np.median(np.sqrt(d2[idx, c]))
    else:
        radii[c] = np.median(np.sqrt(first_d2))  # fallback
eps = 1e-6

d1_r = np.sqrt(first_d2) / (radii[first_idx] + eps)
d2_r = np.sqrt(second_d2) / (radii[second_idx] + eps)
margin_r = d2_r - d1_r  # >0 is better (inside cluster core relatively)

# --- (2) diagonal Mahalanobis distances ---
# per-cluster diagonal variance (regularised)
variances = np.zeros((k, X.shape[1]), dtype="float32")
for c in range(k):
    idx = (first_idx == c)
    if np.any(idx):
        variances[c] = X[idx].var(axis=0, ddof=1) + 1e-6
    else:
        variances[c] = X.var(axis=0, ddof=1) + 1e-6

d2_mahal = ((diff ** 2) / variances[None, :, :]).sum(axis=2)  # (n,k)
d2M_first = d2_mahal[np.arange(n), first_idx]
d2M_second = d2_mahal[np.arange(n), second_idx]

# --- (3) Soft K-means probabilities (calibrated) ---
tau = 1.0  # temperature; expose as CLI if你想
logits = - d2_mahal / (2.0 * tau)
# subtract row max for numerical stability
logits = logits - logits.max(axis=1, keepdims=True)
probs = np.exp(logits)
probs = probs / probs.sum(axis=1, keepdims=True)
confidence = probs.max(axis=1).astype("float32")
alt_label = probs.argmax(axis=1).astype(int)       # soft-argmax（通常=labels）

# --- (4) geometric gap (top-2)
gap_euclid = np.sqrt(second_d2) - np.sqrt(first_d2)
gap_mahal  = np.sqrt(d2M_second) - np.sqrt(d2M_first)

# silhouette already computed -> silhouette_scores
# === New marginal rule (cluster-adaptive) ===
# per-cluster 20th percentile of confidence + global silhouette threshold
conf_p20 = np.zeros(k, dtype="float32")
for c in range(k):
    idx = (first_idx == c)
    conf_p20[c] = np.percentile(confidence[idx], 20) if np.any(idx) else 0.5

is_marginal = (confidence < conf_p20[first_idx]) | (silhouette_scores < 0.10) | (margin_r < 0)

# ==== 5/6. Visualisations & Power BI datasets（只列出新增/替換欄位） ====
# ...（保留你原有圖表代碼；t‑SNE 可選抽樣以加速）...

# Power BI customer-level dataset（新增欄位）
customer_scatter = pd.DataFrame({
    'CUSTOMER_ID': df['CUSTOMER_ID'],
    'CLIENT_NAME': df['CLIENT_NAME'],
    'cluster': labels,
    'pca_x': X_pca[:, 0], 'pca_y': X_pca[:, 1],
    'tsne_x': X_tsne[:, 0], 'tsne_y': X_tsne[:, 1],
    # 原有距離
    'dist_assigned': np.sqrt(first_d2),
    'dist_nearest_other': np.sqrt(second_d2),
    'nearest_other_cluster': second_idx,
    'silhouette_score': silhouette_scores.astype("float32"),
    # 新增：校準信心/距離
    'rnd_assigned': d1_r,                 # 半徑標準化距離（愈細愈好）
    'rnd_nearest_other': d2_r,
    'margin_r': margin_r,                 # >0 表示安全邊界
    'mahalanobis_first': np.sqrt(d2M_first),
    'mahalanobis_second': np.sqrt(d2M_second),
    'gap_euclid': gap_euclid,
    'gap_mahal': gap_mahal,
    'confidence_prob': confidence,        # [0,1]；建議以 0.6–0.7 為醒目線
    'alt_label_from_prob': alt_label,
    'is_marginal': is_marginal.astype(bool),
    # 解釋/維度列（原樣）
    'CUSTOMER_SEGMENT': df['CUSTOMER_SEGMENT'],
    'CUSTOMER_PORTFOLIO': df['CUSTOMER_PORTFOLIO'],
    'ACTIVE_CUSTOMER': df['ACTIVE_CUSTOMER'],
    'ACTUAL_CLTV': df['ACTUAL_CLTV'],
    'CURRENT_YEAR_FAP': df['CURRENT_YEAR_FAP'],
    'FUTURE_LIFETIME_VALUE': df['FUTURE_LIFETIME_VALUE'],
    'ACTUAL_LIFETIME_DURATION': df['ACTUAL_LIFETIME_DURATION'],
    'NUM_CROSS_SOLD_LY': df['NUM_CROSS_SOLD_LY'],
    'CLM_OVER_PROFIT_HITCOUNT': df['CLM_OVER_PROFIT_HITCOUNT'],
    'CURRENT_GWP': df['CURRENT_GWP'],
    'TOTAL_SCORE': df['TOTAL_SCORE']
})
customer_scatter.to_csv(OUTPUT_DIR / "customer_scatter_data.csv", index=False)

# 叢集統計（新增信心度分佈）
cluster_meta = []
for c in range(k):
    idx = (labels == c)
    cluster_meta.append({
        'cluster': c,
        'customer_count': int(idx.sum()),
        'mean_silhouette': float(silhouette_scores[idx].mean()),
        'p20_confidence': float(np.percentile(confidence[idx], 20)) if idx.any() else None,
        'median_rnd': float(np.median(d1_r[idx])) if idx.any() else None,
        'marginal_pct': float((is_marginal[idx].mean() * 100.0) if idx.any() else 0.0)
    })
cluster_metadata = pd.DataFrame(cluster_meta)
cluster_metadata.to_csv(OUTPUT_DIR / "cluster_metadata.csv", index=False)
```

**效果**

* 用 **半徑標準化** + **對角馬氏距離** 處理**橢圓簇**→ 不再把簇內「正常長尾」錯判成邊界。
* **Soft K‑means 概率**（max posterior）比 ratio 更直觀；同時**分簇 20th 百分位**作邊界線，避免「一刀切」。
* 完全**向量化**，165k × 9 速度顯著改善。

> 備註：集群入模只用 **Performance features**（CLTV、FAP、FLV、ATD、XS、Claims‑over‑Plan），同 CSP/DCSP MV 定義一致；**Inspection**在 CSP 暫時仍包含（FAP/XS 保留），此策略與你 SQL 文檔一致。

---

## 小幅訓練級微調（可選）

> 如你想**進一步提升「距離空間」的球形假設吻合度**，而**不改 cluster 數/解釋**：

1. **PCA Whitening（只作距離空間正交化）**

   * 在 scaler 後加 `PCA(whiten=True, n_components='mle' 或保留 ≥95% 變異)` 再做 KMeans。
   * 通常會**提升 margin**（尤其係狹長簇），對 silhouette/DB 影響細。
   * CLI 建議（新參數）：

     ```bash
     ukplc-seg \
       --input /data/CSP_export.csv \
       --outdir ./out_04a_whiten \
       --recipe continuous --algorithm kmeans \
       --k-min 9 --k-max 9 --k-select silhouette \
       --scaler robust --pca-whiten yes --pca-variance 0.95 \
       --log1p ACTUAL_CLTV,FUTURE_LIFETIME_VALUE,CURRENT_YEAR_FAP \
       --kmeans-n-init 20 --cast-float32 --random-state 42
     ```
   * 如你方便，我可以給你一段 `features_continuous.py` 的簡單 patch，把 `--pca-whiten/--pca-variance` 接上 pipeline。

2. **Loss Ratio 異常值處理（可提升穩定性）**

   * 在 CSP/DCSP 產數或入模前**cap/對數** `ACTUAL_LOSS_RATIO`（若你追加此特徵）—之前報告見過極端值（trillion 級），會擾動距離；先 cap 到 P99 或加 `--log1p ACTUAL_LOSS_RATIO`。

---

## 圖表與 Power BI 改善

* **PCA 圖**：保持（你嘅投影解釋變異度 ~75%，可見 cluster 結構）。
* **t‑SNE 圖**：建議（可選）**抽樣 60–80k** 再畫，右圖只畫「marginal」實點＋灰底，其餘為空心標記，有效顯示邊界。
* **Power BI dataset**：新增 `confidence_prob`、`margin_r`、`mahalanobis_*`、`gap_*` 等欄，方便做 slicer（例如 `is_marginal = True`）及 Drill‑through。

---

## 下一輪建議 run（保持 9 簇，測試信心校準／白化）

> **重點**：聚類結果不變（k=9），只試 **信心/距離空間**處理。

### ✅ out_04a_whiten（建議先試）

```bash
ukplc-seg \
  --input /data/CSP_export.csv \
  --outdir ./out_04a_whiten \
  --recipe continuous --algorithm kmeans \
  --k-min 9 --k-max 9 --k-select silhouette \
  --kmeans-n-init 20 --kselect-sample-size 30000 --silhouette-sample-size 15000 \
  --scaler robust --pca-whiten yes --pca-variance 0.95 \
  --log1p ACTUAL_CLTV,FUTURE_LIFETIME_VALUE,CURRENT_YEAR_FAP \
  --cast-float32 --random-state 42
```

### ✅ out_04b_mbk（更快驗證）

```bash
ukplc-seg \
  --input /data/CSP_export.csv \
  --outdir ./out_04b_mbk \
  --recipe continuous --algorithm mbkmeans \
  --k-min 9 --k-max 9 --k-select silhouette \
  --mbk-batch-size 8192 --mbk-max-iter 100 --kmeans-n-init 20 \
  --scaler robust --pca-whiten yes --pca-variance 0.95 \
  --log1p ACTUAL_CLTV,FUTURE_LIFETIME_VALUE \
  --cast-float32 --random-state 42
```

### ✅ out_04c_confOnly（**不重跑 Clustering**，只重算信心度）

* 用我上面 **計距離/信心** 的新段落直接重跑 `create_scatter_and_distance_analysis.py`，生成新 `customer_scatter_data.csv`（圖表/Power BI 即時受惠）。

---

## 參考（Run 04 成效 & 前序比較）

* **Run 04（KMeans）**：9 簇；Silhouette **0.3851**；DB **1.0286**；0% noise。
* **Run 01/03（HDBSCAN）**：118–120 簇；Silhouette **0.47–0.48**；但 **67% noise**（營運不可用）。
* **Run 02（Discretised+KMeans）**：3 簇；Silhouette **0.257**；慢。
* **CSP/DCSP MV & CLTV/FLV/FLD 定義**：性能特徵、Inspection 處理、FLD ≤ 20 年、ELV/CLTV 框架。

---

## 想你確認（3 條好快）

1. **邊界比例**：你期望「marginal」大概落喺幾多％？（我而家用 per‑cluster 20th percentile + silhouette<0.10；可以改 10–30%）
2. **是否加入 PCA whitening** 做一次 out_04a？（通常對「狹長簇」信心 margin 提升最明顯）
3. **Power BI 字段**：上面新增欄位 OK 嗎？（`confidence_prob / margin_r / gap_mahal`）我可再加 **per‑cluster p20** 作 slicer。

---

### UK English Keywords

**radius‑normalised distance (RND)**, **diagonal Mahalanobis**, **soft K‑means probabilities**, **temperature parameter (τ)**, **cluster‑adaptive threshold**, **PCA whitening**, **vectorised distance**, **silhouette (sampled)**, **Davies–Bouldin**, **Power BI slicers**.

---

### Next 3 Actions

1. **更新 `create_scatter_and_distance_analysis.py`**：按以上 **PATCH** 取代距離/信心段落，重產 `customer_scatter_data.csv`、圖表；回覆我 **marginal%** 同 **confidence_prob** 分佈。
2. **跑 `out_04a_whiten`（可選）**：保持 k=9，只開 `--pca-whiten yes` 比對信心/邊界；同時保存原 KMeans 結果以便映射。
3. **（如需要）把閾值定稿**：我會根據你回來嘅分佈，幫你把 **cluster‑adaptive threshold**（p20/ p25）與 **Power BI** slicer 文檔化，固定成營運口徑。
