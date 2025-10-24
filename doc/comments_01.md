多謝你提供 repo 同三個 run（`out_01`～`out_03`）嘅報告。我已經用你上載嘅 **01/02/03 REPORT + CONFIGURATION_REPORT** 同埋 `cluster_profiles.csv` 去核對表現同瓶頸，以下係**診斷 + 微調（fine‑tune）建議**，以及**下一輪建議配置**（可以直接跑）。
（我亦嘗試打開你提供嘅 GitHub 連結，不過瀏覽器層面未能取回頁面；幸好你已上載完整報告檔，我用嗰啲做準則。）

---

## 先講重點（TL;DR）

* `out_01` / `out_03`（**continuous + auto→HDBSCAN**）：分離度指標**好到非常好**（Silhouette ≈ **0.47–0.48**；DB ≈ **0.94–0.95**），**但有 ~67% 噪音（cluster = -1）**，而且 cluster 數目 **118–120**，**過份碎**、難以落地營運。
  其對應的配置報告同樣指出**高 Noise 比例（67%）**及建議試 **KMeans** 或降低 `min_cluster_size`。
* `out_02`（**discretised + KMeans，k=3**）：**100% 覆蓋（無 noise）**、簇形簡單可解釋，但 **Silhouette≈0.257**（分隔一般），**而你亦反映 discretised recipe 很慢**（165k rows 已覺明顯）。

👉 綜合：**HDBSCAN 太碎 + 高噪音**；**Discretised 太慢 + 分隔一般**。我建議用 **Continuous +（MiniBatch）KMeans** 作主線，**Discretised 改為 OHE(sparse) + SVD + MBKMeans** 作對照；所有內部評分改用**抽樣**，大幅提速，同時收窄 k 範圍（例如 6–10）提升穩定性。

---

## 你三次 run 的快速對照

| Run      | Recipe          | Algorithm        | Clusters | Silhouette |        DB | 觀察              |
| -------- | --------------- | ---------------- | -------: | ---------: | --------: | --------------- |
| `out_01` | continuous      | auto→**HDBSCAN** |  **118** |  **0.470** | **0.941** | ~67% noise（-1）。 |
| `out_02` | **discretised** | **KMeans (k=3)** |    **3** |      0.257 |     1.919 | 簇大而闊、可解釋、速度慢。   |
| `out_03` | continuous      | auto→**HDBSCAN** |  **120** |  **0.476** | **0.946** | ~67% noise（-1）。 |

> HDBSCAN 噪音（-1）約 **111k/165k（~67%）**，KMeans 雖然全覆蓋，但 3 簇分隔偏弱。配置報告亦明確建議試 KMeans / 調低 `min_cluster_size`。

---

## 微調方向（針對你嘅痛點）

### A) **Continuous 主線：KMeans / MiniBatchKMeans（MBKMeans）**

* **縮窄 k 範圍**：`k ∈ [6, 10]`（比 3–12 更穩定）。
* **重尾特徵做 `log1p`**：至少對 `ACTUAL_CLTV`, `FUTURE_LIFETIME_VALUE`,（可選）`CURRENT_YEAR_FAP`。
* **Scaler**：`RobustScaler`（對 outliers 更穩）或 `StandardScaler` 作對照。
* **抽樣評分**：`silhouette_sample_size ≈ 15k`、`kselect_sample_size ≈ 30k`。
* **大數據友好**：如 100k+ 行，用 **MBKMeans**（`batch_size≈8k`、`max_iter≈100`）可大幅提速。

### B) **Discretised 對照：One‑Hot(sparse) + SVD → MBKMeans**

* **One‑Hot** 改 **sparse**，之後加 **TruncatedSVD（≈24 維）**；最後 **MBKMeans**。
* **量化分箱**：`n_bins` 由 7 降到 **5**（減碎片化與噪聲）。
* 全線用 **float32** 減 RAM。

> 上述建議同你 **CSP/DCSP 物料化視圖** 定義一致：**Inspection** 暫時**保留**，但遵守 MV 的計算口徑（例如 Inspection 只保留 FAP、Cross‑sell；其他度量排除）。

---

## 建議下一輪（out_04～out_08）**可直接執行**配置

> 以下假設你沿用我哋之前提供嘅 CLI（`ukplc-seg`）。如你 repo 用另一個入口，可把同等參數映射回去。

### ✅ `out_04` —— **Continuous + KMeans（主推）**

```bash
ukplc-seg \
  --input /data/CSP_export.csv \
  --outdir ./out_04 \
  --recipe continuous \
  --algorithm kmeans \
  --k-min 6 --k-max 10 --k-select silhouette \
  --kmeans-n-init 20 \
  --kselect-sample-size 30000 --silhouette-sample-size 15000 \
  --scaler robust \
  --log1p ACTUAL_CLTV,FUTURE_LIFETIME_VALUE,CURRENT_YEAR_FAP \
  --cast-float32 --random-state 42
```

**目的**：取代 HDBSCAN 的高噪音；保持良好分隔但簇數落地。

---

### ✅ `out_05` —— **Continuous + MBKMeans（更快對照）**

```bash
ukplc-seg \
  --input /data/CSP_export.csv \
  --outdir ./out_05 \
  --recipe continuous \
  --algorithm mbkmeans \
  --k-min 6 --k-max 10 --k-select silhouette \
  --kmeans-n-init 20 --mbk-batch-size 8192 --mbk-max-iter 100 \
  --kselect-sample-size 30000 --silhouette-sample-size 15000 \
  --scaler standard \
  --log1p ACTUAL_CLTV,FUTURE_LIFETIME_VALUE \
  --cast-float32 --random-state 42
```

**目的**：大幅提速；與 out_04 作 scaler 對照。

---

### ✅ `out_06` —— **Discretised（快速版）+ SVD + MBKMeans**

```bash
ukplc-seg \
  --input /data/CSP_export.csv \
  --outdir ./out_06 \
  --recipe discretised \
  --n-bins 5 --svd-components 24 \
  --algorithm mbkmeans \
  --k-min 6 --k-max 10 --k-select silhouette \
  --kmeans-n-init 20 --mbk-batch-size 8192 --mbk-max-iter 100 \
  --kselect-sample-size 30000 --silhouette-sample-size 15000 \
  --cast-float32 --random-state 42
```

**目的**：解決你提到嘅 **discretised 很慢**（165k rows）— 改用 **OHE(sparse)→SVD**，速度會明顯改善，同時保留 case‑study 式「分箱→聚類」可解讀性。

---

### 🔎 `out_07`（選擇性）—— **DCSP（Digital）同配方對照**

```bash
ukplc-seg \
  --input /data/DCSP_export.parquet \
  --outdir ./out_07 \
  --recipe continuous \
  --algorithm kmeans \
  --k-min 5 --k-max 9 --k-select silhouette \
  --kmeans-n-init 20 \
  --kselect-sample-size 30000 --silhouette-sample-size 15000 \
  --scaler robust \
  --log1p ACTUAL_CLTV,FUTURE_LIFETIME_VALUE \
  --cast-float32 --random-state 42
```

**目的**：CSP / DCSP 結構相同但行為差異，分別優化可提升部署質素。

---

### 🧪 `out_08`（如仍想試 HDBSCAN）—— 降噪音設定

> 只建議**抽樣**先做概念驗證（例如 60k 行），因為 165k 用 HDBSCAN 會慢，而且噪音高。

```bash
ukplc-seg \
  --input /data/CSP_export.csv \
  --outdir ./out_08 \
  --recipe continuous \
  --algorithm hdbscan \
  --hdbscan-min-cluster-size 50 --hdbscan-min-samples 10 \
  --scaler robust \
  --log1p ACTUAL_CLTV,FUTURE_LIFETIME_VALUE \
  --cast-float32 --random-state 42 \
  --sample 60000
```

**目的**：以較細 `min_cluster_size` / `min_samples` 減低 -1 噪音，同時避免過度碎片化（先抽樣驗證）。你之前 HDBSCAN 噪音 ~67% 已在報告記錄。

---

## Code‑level 改善（可直接抄入你 repo）

> 你提到已修正少量 glitches；以下改動**向後兼容**，只係加參數/步驟。

### 1) **Discretised 快速版**（OHE→SVD→MBKMeans）

```python
# features_discretised.py
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import MiniBatchKMeans

NUM_FEATURES = [
    "ACTUAL_CLTV","CURRENT_YEAR_FAP","FUTURE_LIFETIME_VALUE",
    "ACTUAL_LIFETIME_DURATION","NUM_CROSS_SOLD_LY","CLM_OVER_PROFIT_HITCOUNT"
]

def build_discretised_svd_mbkmeans(n_bins=5, svd_components=24, k=8,
                                   batch_size=8192, max_iter=100, random_state=42):
    kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    svd = TruncatedSVD(n_components=svd_components, random_state=random_state)
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size,
                             max_iter=max_iter, n_init=20, random_state=random_state)
    pipe = Pipeline([
        ("kbins", kb),
        ("ohe", ohe),
        ("svd", svd),
        ("cluster", kmeans),
    ])
    return ColumnTransformer([("numerics", pipe, NUM_FEATURES)], remainder="drop")
```

### 2) **Continuous 主線**（KMeans / MBKMeans + log1p + Robust）

```python
# features_continuous.py
import numpy as np
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, MiniBatchKMeans

NUM_FEATURES = [
    "ACTUAL_CLTV","CURRENT_YEAR_FAP","FUTURE_LIFETIME_VALUE",
    "ACTUAL_LIFETIME_DURATION","NUM_CROSS_SOLD_LY","CLM_OVER_PROFIT_HITCOUNT"
]

def build_continuous_cluster(k=8, algorithm="kmeans", scaler="robust",
                             log1p_cols=("ACTUAL_CLTV","FUTURE_LIFETIME_VALUE","CURRENT_YEAR_FAP"),
                             random_state=42, batch_size=8192, max_iter=100):
    def _log1p_df(X):
        X = X.copy()
        for c in log1p_cols:
            if c in X:
                X[c] = np.log1p(np.clip(X[c].astype("float32"), a_min=0, a_max=None))
        return X

    scaler_step = RobustScaler() if scaler == "robust" else StandardScaler()
    cluster = (MiniBatchKMeans(n_clusters=k, batch_size=batch_size, max_iter=max_iter, n_init=20,
                               random_state=random_state)
               if algorithm == "mbkmeans" else
               KMeans(n_clusters=k, n_init=20, random_state=random_state))
    pipe = Pipeline([
        ("log1p", FunctionTransformer(_log1p_df, validate=False)),
        ("scaler", scaler_step),
        ("cluster", cluster),
    ])
    return ColumnTransformer([("numerics", pipe, NUM_FEATURES)], remainder="drop")
```

### 3) **評分抽樣以提速**

```python
# metrics.py
from sklearn.metrics import silhouette_score
import numpy as np

def silhouette_on_sample(X, labels, sample_size=15000, random_state=42):
    n = X.shape[0]
    if n > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=sample_size, replace=False)
        return silhouette_score(X[idx], labels[idx])
    return silhouette_score(X, labels)
```

> 上述改動正正回應你兩個痛點：
> ① **discretised 很慢** → 改用 **OHE(sparse) + SVD + MBKMeans**；
> ② **HDBSCAN 噪音太多** → 以 **（MiniBatch）KMeans** 配合 `log1p + Robust` 提升分隔度同落地性。
> （同時保留 **Inspection** 嘅處理口徑：FAP/Cross‑sell 計入，其他度量排除—見 CSP/DCSP MV 文檔。）

---

## 跑完之後請回傳/覆述以下（方便我再細調）

1. **`internal_metrics.json`**：`n_clusters`、`silhouette`、`calinski_harabasz`、`davies_bouldin`。
2. **`cluster_profiles.csv`**：最大/最小簇大小、`High:Low` 比（避免極端不均）。
3. **`cluster_vs_existing_segment.csv`**：新簇同 **A/A+/B/C/D/E/F** 映射（跨渠道/Inspection 口徑一致）。

> 參考：你現有 HDBSCAN 兩次 run 嘅配置報告已指出 **67% 噪音**，建議比較 KMeans；discretised+k=3 的簡潔分割與慢速亦已在報告記錄。

---

##（可選）商務落地小貼士

* **簇解釋（Explanatory metrics）**：在 `cluster_profiles.csv` 報表中，用 **Product Index**、**Leverage（CLTV/客）**、**Loss ratio**、**Active %** 去定義「**留存／交叉銷售／VIP 保護**」動作（你現有報告已展示類似欄位）。
* **渠道分割**：CSP / DCSP 保持**互斥 hard‑cut**（客層級），避免重覆計數—見 MV 文檔。

---

## 需要你回覆的 3 條短問題

1. 之後你傾向以 **Continuous +（MiniBatch）KMeans** 作**主**，而 **Discretised（SVD 版）**作**對照**可以嗎？
2. `k` 初步用 **6–10** 可以？如你想更 granular，我可以追加 **k=5/7/9/11** 的 sweep。
3. DCSP（Digital）你會跟住跑嗎？若係，我會把 `k` 範圍略為收窄（例如 **5–9**）以提穩定性。

---

## 參考（你已上載的報告/文檔）

* `out_03` 報告：continuous + auto（HDBSCAN），**120** clusters，Silhouette **0.4757**。
* `out_02` 報告：discretised + KMeans，**k=3**，Silhouette **0.2570**。
* `out_01` 報告：continuous + auto（HDBSCAN），**118** clusters，Silhouette **0.4703**。
* `out_03`/`out_01` 配置報告（均指出 **~67% 噪音**，建議試 KMeans / 調參）：
* CSP/DCSP MV（Inspection 處理、Cross‑sell 維度、Hard‑cut）：

---

### UK English Keywords

**MiniBatchKMeans**, **TruncatedSVD**, **quantile binning**, **RobustScaler**, **silhouette (sampled)**, **Calinski–Harabasz**, **Davies–Bouldin**, **cross‑sell (Product Index)**, **Inspection handling**, **A/B/C/D/E/F segments**, **hard‑cut portfolio**, **ELV/FLV/CLTV**, **out‑of‑time**.

---

### Next 3 Actions

1. 先跑 **`out_04`**（Continuous + KMeans）及 **`out_05`**（MBKMeans 對照）；如時間可行，再跑 **`out_06`**（Discretised SVD 快速版）。
2. 把三個 run 的 **`internal_metrics.json`**、**`cluster_profiles.csv`**（最大/最小簇同比例）、同 **`cluster_vs_existing_segment.csv`** 重點數字回覆我。
3. 告訴我你是否同時要跑 **DCSP**；我會基於結果再收窄 `k`、調整 `svd-components` / `batch size`，並可提供一個小型 **patch** 直接合併你 repo。