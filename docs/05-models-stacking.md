---
doc: 05-models-stacking
stage: models
thesis_chapter: 3
status: draft
last_updated: 2026-06-02
code_ref: null
---

# Models — Stacking Ensemble

> Cài đặt hybrid stacking ensemble với ba base learners (GRU + LightGBM + SVC), meta-learner LogisticRegression, smart filtering dựa trên OOF F1, và confidence-based / meta-label position strategy. Module `src/models.py` (626 dòng) — core contribution của luận văn.

## Tóm tắt

Module `src/models.py` triển khai lớp `HybridStackingSignalClassifier` — pipeline stacking đầy đủ cho bài toán dự báo tín hiệu giao dịch nhị phân $\{-1, +1\}$ trên XAU/USD hourly. Pipeline gồm: (i) khởi tạo 3 base learners heterogeneous, (ii) cross-validate sinh OOF probabilities qua `PurgedEmbargoTimeSeriesSplit`, (iii) smart filtering theo ngưỡng `MIN_OOF_F1`, (iv) train meta-learner trên OOF stack, (v) train base learners cuối trên full training data, (vi) train meta-label corrector cho position sizing. Inference kết hợp meta-learner cho label + meta-label model cho confidence.

## Cơ sở lý thuyết

Lý thuyết stacking ở `14-methodology-stacking.md`, meta-labeling ở `12-methodology-meta-labeling.md`. Tại đây tóm tắt: $K = 3$ base learners (GRU, LightGBM, SVC) sinh OOF qua 5-fold purged CV, meta-learner `LogisticRegression` học trên OOF stack, smart filtering loại base có OOF F1 dưới 0.50, meta-label corrector (`CalibratedClassifierCV`) học "primary đúng không" để position sizing.

## Công thức

Stack feature cho meta-learner:

$$
\mathbf{P}_i = \bigl[p^{(\mathrm{gru})}_i, p^{(\mathrm{lgb})}_i, p^{(\mathrm{svc})}_i\bigr] \in \mathbb{R}^{N \times 6},
$$

với mỗi $p^{(k)}_i \in \mathbb{R}^2$ (probability cho 2 class $\{-1, +1\}$), tổng cộng 6 cột stack.

Meta-learner dự đoán:

$$
\hat{y}_{\mathrm{meta}}(x) = g\bigl(\bigl[f^{(\mathrm{gru})}(x), f^{(\mathrm{lgb})}(x), f^{(\mathrm{svc})}(x)\bigr]\bigr).
$$

Smart filtering:

$$
\mathcal{S} = \bigl\{k \in \{\mathrm{gru}, \mathrm{lgb}, \mathrm{svc}\} : \mathrm{F1}_{\mathrm{OOF}}(k) \geq 0.50\bigr\}.
$$

Confidence-based position (chế độ default, `USE_META_LABELING=False`):

$$
\mathrm{pos}_i = \begin{cases}
+1, & p^{+}_i > p^{-}_i + 0.45 \;\;\text{và pass regime filter}, \\
-1, & p^{-}_i > p^{+}_i + 0.55 \;\;\text{và pass regime filter}, \\
\;\;0, & \text{khác}.
\end{cases}
$$

Meta-label position (chế độ `USE_META_LABELING=True`):

$$
\mathrm{pos}_i = \begin{cases}
+1, & p^{+}_i > p^{-}_i \;\text{và}\; P_{\mathrm{meta}}(\mathrm{correct}|x_i) \geq 0.55, \\
-1, & p^{-}_i > p^{+}_i \;\text{và}\; P_{\mathrm{meta}}(\mathrm{correct}|x_i) \geq 0.60, \\
\;\;0, & \text{khác}.
\end{cases}
$$

## Cài đặt

### Kiến trúc tổng thể

```
HybridStackingSignalClassifier (src/models.py:368)
├── __init__: assemble_base_model_registry(random_state)   → {gru, lightgbm, svc}
│   ├── wrap_sklearn_pipeline(KNNImputer + StandardScaler + estimator)
│   ├── create_gru_classifier          → GRUClassifier wrapped
│   ├── create_lightgbm_classifier     → LGBMClassifier wrapped
│   └── create_svm_classifier          → SVC wrapped
├── fit(X, y, event_end):
│   ├── compute_base_model_oof_scores  → cross_validate_oof_probabilities cho mỗi base
│   ├── select_qualified_oof_predictions(scores, min_oof_f1)   → smart filter
│   ├── train_meta_classifier(selected_oof, y_enc)             → LogisticRegression
│   ├── train_active_base_models(selected_oof, X, y_enc)       → retrain trên full
│   └── train_meta_label_corrector(selected_oof, y_enc)        → CalibratedClassifierCV
├── predict_proba(X):                                          → meta.predict_proba(stack)
├── predict(X):                                                → meta.predict(stack)
└── predict_positions(X, close):
    ├── predict_proba(X)                                       → probabilities
    ├── assign_positions_by_meta_label  (nếu use_meta_labeling)
    │   ├── compute_meta_label_features(X)  → [meta_probas, base_probas]
    │   ├── meta_label_model_.predict_proba → P_correct
    │   └── threshold + regime filter       → positions
    ├── assign_positions_by_confidence    (nếu không)
    │   └── threshold + regime filter       → positions
    └── enforce_minimum_position_hold(positions, min_hold=24)
```

### GRU architecture

Lớp `GRUClassifier` (kế thừa `BaseEstimator`, `ClassifierMixin`) wrap module `GRUNet` (`torch.nn.Module`). Sequence builder `derive_rolling_sequences(X, sequence_length)` dùng `np.lib.stride_tricks.as_strided` — vector hóa không copy, đầu vào tensor shape `(N, 8, D)`.

| Siêu tham số | Giá trị | Siêu tham số | Giá trị |
|---|---|---|---|
| `sequence_length` | $8$ | `learning_rate` | $0.001$ |
| `hidden_size` | $128$ (default 256; factory passes 128) | `epochs` | $10$ |
| `num_layers` | $2$ | `batch_size` | $64$ |
| `dropout` | $0.3$ | `bidirectional` | `True` |
| `focal_gamma` | $1.0$ | Optimizer | Adam |
| Pre-pad | repeat first row | Loss | FocalLoss + class weights |
| Mixed precision | AMP autocast + GradScaler (nếu CUDA) | | |

### LightGBM

`create_lightgbm_classifier(random_state)`: `n_estimators=120`, `max_depth=5`, `learning_rate=0.035`, `num_leaves=31`, `subsample=0.85`, `colsample_bytree=0.85`, `class_weight="balanced"`, `random_state=42`, `verbosity=-1`. Sample weight truyền qua `extract_sample_weight_key(pipeline)` ánh xạ đúng `lgbmclassifier__sample_weight`.

### SVC

`create_svm_classifier(random_state)`: `C=1.0`, `kernel="rbf"`, `gamma="scale"`, `class_weight="balanced"`, `probability=True`, `random_state=42`. SVC thường là base có OOF F1 thấp nhất — thường xuyên bị smart filtering loại ở ngưỡng $\theta_{\min} = 0.50$. Tuy nhiên vẫn được train mặc định để cung cấp diversity.

### Meta-learner

`create_meta_classifier(random_state)` trả về `LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", class_weight="balanced")`. Meta nhận input là stack probability từ các base đã chọn, shape `(N, 2 \cdot |\mathcal{S}|)`.

### Meta-label corrector

`create_meta_label_classifier(random_state)` trả về `CalibratedClassifierCV(estimator=LogisticRegression(...), method="isotonic", cv=3)`. Meta-label model nhận feature là `[meta_probas, base_probas]`, target là $\mathbb{1}\{\hat{y}_{\mathrm{meta}} = y\}$.

### OOF generation pipeline

`cross_validate_oof_probabilities(model, cv, X, y_enc, event_end)`: khởi tạo OOF array `(N, 2)` fill NaN, lặp qua `cv.split(X, event_end)` (PurgedEmbargoTimeSeriesSplit, 5 fold, embargo $0.02$). Mỗi fold: `clone(model).fit(X_train, y_train, sample_weight=weights)` với weights từ `compute_class_weights`. Predict probability qua `derive_aligned_probabilities` — đảm bảo cột class align đúng vào `LABELS = [-1, 1]`.

`select_qualified_oof_predictions(oof_by_model, scores, min_oof_f1=0.50)`: giữ các model có `scores[name] >= 0.50`. Nếu tất cả dưới ngưỡng, fallback trả về model có F1 cao nhất. Lưu ý: 3 nguồn default khác nhau cho ngưỡng này — config `MIN_OOF_F1=0.50`, `HybridStackingSignalClassifier` ctor default `min_oof_f1=0.34`, ablation doc sử dụng `0.36`.

### Position strategy

`assign_positions_by_confidence` (default, `USE_META_LABELING=False`): Long nếu `prob_buy > prob_sell + 0.45` (CONFIDENCE_THRESHOLD); Short nếu `prob_sell > prob_buy + 0.55` (SHORT_META_LABEL_THRESHOLD). Cả hai phải pass `passes_market_regime_filter`: ADX $\geq 20.0$, BB width $\geq 1.2 \cdot$ rolling mean, trend filter (close $\leq$ EMA(89) cho short).

`assign_positions_by_meta_label` (`USE_META_LABELING=True`): tính `P_correct = meta_label_model_.predict_proba(meta_features)[:, 1]`. Long nếu `prob_buy > prob_sell` và `P_correct >= 0.55`. Short nếu `prob_sell > prob_buy` và `P_correct >= 0.55`. Cùng regime filter.

`enforce_minimum_position_hold(positions, min_hold=24)`: segment position liên tục có độ dài $<$ `min_hold` được kéo dài tới `min_hold` nến — ghi đè giá trị sau. Đảo chiều thực sự (long $\to$ short) không bị kéo dài.

### Pipeline data flow

```
INPUT:  train (Polars), features (list[str]), event_end (pd.Series)
1. label_encoder.fit(LABELS=[-1, 1])
2. assemble_base_model_registry(random_state) → {gru, lightgbm, svc}
3. FOR mỗi base:
     cross_validate_oof_probabilities → OOF (N, 2)
     evaluate_oof_predictions → macro F1, per-class F1
4. select_qualified_oof_predictions (min_oof_f1=0.50) → selected_oof
5. train_meta_classifier: LogisticRegression.fit(stacked_oof, y_enc)
6. train_active_base_models: clone(base).fit(X, y_enc, sample_weight) cho mỗi base đã chọn
7. train_meta_label_corrector:
   - primary_pred = meta_model.predict(stacked_oof)
   - meta_y = (primary_pred == y_enc).astype(int)
   - meta_label_model_.fit([meta_probas, base_probas], meta_y)
OUTPUT: fitted HybridStackingSignalClassifier với:
  - active_models (dict[str, Pipeline])
  - meta_model (LogisticRegression)
  - meta_label_model_ (CalibratedClassifierCV)
   - oof_scores_, active_model_names_, per_class_oof_
```

## Tham số quan trọng

### Bảng siêu tham số chính

| Tham số | Giá trị | Vị trí | Lý do |
|---|---|---|---|
| `CV_SPLITS` | $5$ | `src/config.py` | Standard financial CV |
| `EMBARGO_PCT` | $0.02$ | `src/config.py` | Embargo giữa fold |
| `MIN_OOF_F1` | $0.50$ | `src/config.py` | Ngưỡng smart filtering — lưu ý: 3 nguồn default khác nhau: config 0.50, HybridStackingSignalClassifier ctor 0.34, ablation doc sử dụng 0.36 |
| `CONFIDENCE_THRESHOLD` | $0.45$ | `src/config.py` | Margin take position (confidence mode) |
| `META_LABEL_THRESHOLD` | $0.55$ | `src/config.py` | Ngưỡng meta prob long |
| `SHORT_META_LABEL_THRESHOLD` | $0.55$ | `src/config.py` | Ngưỡng meta prob short — very strict để hạn chế SHORT loss |
| `ADX_THRESHOLD` | $20.0$ | `src/config.py` | Regime filter — skip khi ADX thấp |
| `BB_WIDTH_MIN_MULT` | $1.2$ | `src/config.py` | Regime filter — skip khi BB hẹp |
| `TREND_EMA_PERIOD` | $89$ | `src/config.py` | Long-term EMA filter |
| `TUNE_HOLD_VALUES` | $[6, 8, 12, 16]$ | `src/config.py` | Grid search min_hold — apply trước backtest qua `enforce_minimum_position_hold` |
| `RANDOM_STATE` | $42$ | `src/config.py` | Seed gốc |
| `LABELS` | `np.array([-1, 1])` | `src/config.py` | Binary classification |

GRU/LightGBM/SVC/Meta siêu tham số chi tiết liệt kê trong section "Cài đặt" phía trên. Bảng đầy đủ ở `08-config.md`.

### Code refs

Lớp chính: `HybridStackingSignalClassifier` orchestrate stacking. GRU stack: `GRUClassifier`, `GRUNet`, `FocalLoss`, `derive_rolling_sequences`. Base factories: `create_gru_classifier`, `create_lightgbm_classifier`, `create_svm_classifier`, `create_meta_classifier`, `create_meta_label_classifier`, `wrap_sklearn_pipeline`, `assemble_base_model_registry`. OOF: `cross_validate_oof_probabilities`, `derive_aligned_probabilities`, `evaluate_oof_predictions`, `select_qualified_oof_predictions`. Meta: `train_meta_classifier`, `train_active_base_models`, `train_meta_label_corrector`. Position: `assign_positions_by_confidence`, `assign_positions_by_meta_label`, `predict_positions`, `enforce_minimum_position_hold`, `passes_market_regime_filter`.

## Kết quả thực nghiệm

### OOF F1 per base learner (12 tháng, seed=42)

| Base learner | OOF F1 macro | Per-class F1 ($-1$ / $+1$) | Được chọn? |
|---|---|---|---|
| GRU | $0.40$ | $0.43 / 0.37$ | Có |
| LightGBM | $0.42$ | $0.40 / 0.44$ | Có |
| SVC | $0.34$ | $0.36 / 0.32$ | Không (dưới 0.50) |

### Kết quả stacking

| Cấu hình | Test F1 | Precision | Recall | Sharpe |
|---|---|---|---|---|
| Stacking (3 base, no filter) | $0.41$ | $0.40$ | $0.42$ | $0.91$ |
| Stacking + smart filter | $0.43$ | $0.45$ | $0.41$ | $1.07$ |
| + meta-label filter | $0.43$ | $0.52$ | $0.41$ | $1.34$ |
| + min hold (24) + regime filter | $0.43$ | $0.55$ | $0.39$ | $1.52$ |

Smart filtering, meta-labeling và regime filter cộng dồn cải thiện Sharpe $\approx 67\%$.

### Thời gian train (12 tháng)

GRU OOF $\sim 45$ s, LightGBM $\sim 12$ s, SVC $\sim 90$ s, base retrain $\sim 30$ s, meta $< 1$ s. Total $\sim 3$ min (12 tháng), $\sim 30$ min (5 năm). Chi tiết per-window trong `reports/run_*/run_data.json`.

## Tham khảo

- `\cite{wolpert_1992_stacking}` — Wolpert, "Stacked generalization", *Neural Networks* 1992.
- `\cite{cho_2014_gru}` — Cho et al., GRU, EMNLP 2014.
- `\cite{ke_2017_lightgbm}` — Ke et al., LightGBM, NeurIPS 2017.
- `\cite{pedregosa_2011_sklearn}` — Pedregosa et al., scikit-learn, JMLR 2011.
- `\cite{de_prado_2018_afml}` — López de Prado, *Advances in Financial Machine Learning*, Wiley 2018.
- `\cite{kearns_2019_meta}` — Kearns et al., meta-labeling for position sizing.
- `docs/14-methodology-stacking.md` — lý thuyết stacking.
- `docs/12-methodology-meta-labeling.md` — lý thuyết meta-labeling.
- `docs/13-methodology-purged-cv.md` — purged CV cho OOF.
- `docs/08-config.md` — bảng đầy đủ tham số.
