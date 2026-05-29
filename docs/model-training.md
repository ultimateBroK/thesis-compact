# Model Training — Hybrid Stacking Signal Classifier

## Mục đích

Huấn luyện **HybridStackingSignalClassifier** — một stacking ensemble kết hợp 3 base models khác nhau (GRU, LightGBM, SVC) với smart filtering và meta-learner.

## Kiến trúc tổng thể

```mermaid
flowchart TD
    subgraph "1. Base Models"
        A1["GRU<br/>PyTorch<br/>sequence_length=8<br/>hidden_size=128"]
        A2["LightGBM<br/>n_estimators=120<br/>max_depth=5"]
        A3["SVC<br/>kernel=rbf<br/>probability=True"]
    end

    subgraph "2. Preprocessing Pipeline"
        B1["KNNImputer<br/>n_neighbors=5"]
        B2["StandardScaler"]
    end

    subgraph "3. OOF Training"
        C["PurgedEmbargoCV<br/>5 folds"]
        D["OOF Predictions<br/>per model"]
    end

    subgraph "4. Smart Filtering"
        E["Score OOF macro F1"]
        F{"F1 >= 0.36?"}
        G["Keep for stacking"]
        H["Filter out"]
    end

    subgraph "5. Meta-Learning"
        I["Stack OOF probas"]
        J["CalibratedClassifierCV<br/>(LogisticRegression, isotonic)"]
    end

    subgraph "6. Position Sizing"
        K["Confidence threshold<br/>0.35 + ADX/BB filter"]
        L["Meta-labeling<br/>(enabled)"]
    end

    X["Train Data<br/>features + labels"] --> A1
    X --> A2
    X --> A3

    A1 --> B1
    A2 --> B1
    A3 --> B1
    B1 --> B2
    B2 --> C
    C --> D
    D --> E
    E --> F
    F --> G
    F -->|"fallback nếu all fail<br/>lấy best model"| H
    G --> I
    H --> I
    I --> J
    J --> K
    J --> L

    style A1 fill:#c084fc,stroke:#e9d5ff
    style A2 fill:#c084fc,stroke:#e9d5ff
    style A3 fill:#c084fc,stroke:#e9d5ff
    style J fill:#60a5fa,stroke:#93c5fd
    style F fill:#34d399,stroke:#6ee7b7
```

## Chi tiết từng thành phần

### 1. Base Models

#### GRU (Gated Recurrent Unit) — `src/models/gru.py:GRUClassifier`

```mermaid
flowchart LR
    A["Sequence 8h<br/>(batch_size=64)"] --> B["GRU<br/>3 layers<br/>hidden_size=256<br/>dropout=0.3<br/>bidirectional=True"]
    B --> C["Last hidden state<br/>AMP mixed precision"]
    C --> D["Linear → 2 classes"]
    D --> E["Focal Loss<br/>gamma=1.0<br/>class_weight=balanced"]
    E --> F["Adam<br/>lr=0.001<br/>epochs=20"]
```

- **Input**: sequences 8 nến 1h (8 bước thời gian)
- **GRU**: 3 layers, hidden size 256, dropout 0.3, bidirectional=True, AMP mixed precision
- **Loss function**: Focal Loss — giảm trọng số các sample dễ, tập trung vào sample khó (hiệu quả với class imbalance)
- **Optimizer**: Adam, lr=0.001, epochs=20
- **Device**: auto CUDA nếu có GPU

#### LightGBM — `src/models/builders.py:create_lightgbm_classifier`

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 120 | Đủ để học patterns, không overfit |
| `max_depth` | 5 | Giới hạn độ sâu |
| `learning_rate` | 0.035 | Step size nhỏ |
| `num_leaves` | 31 | Leaf-wise tree |
| `subsample` | 0.85 | Bagging fraction |
| `colsample_bytree` | 0.85 | Feature fraction |
| `class_weight` | balanced | Xử lý imbalance |
| `verbosity` | -1 | Silent |

#### SVC — `src/models/builders.py:create_svm_classifier`

| Parameter | Value |
|---|---|
| `C` | 1.0 |
| `kernel` | rbf |
| `gamma` | scale |
| `class_weight` | balanced |
| `probability` | True (cần predict_proba cho stacking) |

### 2. OOF (Out-of-Fold) Training

```mermaid
sequenceDiagram
    participant Pipeline as Preprocessing
    participant CV as PurgedEmbargoCV
    participant Model as Base Model

    Pipeline->>CV: X, y, event_end
    loop 5 folds
        CV->>Model: train_idx, val_idx
        Model->>Model: clone() + fit(train_X, train_y)
        Model->>Model: predict_proba(val_X)
        Model-->>CV: OOF predictions
    end
    CV-->>Pipeline: OOF array (N x 2)
```

- Mỗi model được clone và train trên 4/5 folds, predict fold còn lại
- OOF shape: `(N_samples, 2 classes)` — probability distribution
- NaN cho những samples không được predict trong fold nào

### 3. Smart Filtering (`src/models/stacking.py:select_qualified_oof_predictions`)

```python
def select_qualified_oof_predictions(oof_by_model, scores, min_oof_f1):
    selected = {name: oof for name, oof in oof_by_model.items()
                if scores[name] >= min_oof_f1}
    if selected:
        return selected
    best_name = max(scores, key=scores.get)
    return {best_name: oof_by_model[best_name]}
```

- Tính **macro F1** cho OOF predictions của mỗi model
- Chỉ giữ model có F1 >= `MIN_OOF_F1` (0.36)
- Nếu tất cả đều dưới ngưỡng: lấy **model tốt nhất** (fallback)

### 4. Meta-Learner

```mermaid
flowchart LR
    A["OOF probas từ<br/>active models"] --> B["Horizontal stack<br/>concatenate probas"]
    B --> C["LogisticRegression<br/>C=1.0, lbfgs<br/>class_weight=balanced"]
    C --> D["Final classifier"]
```

- Input: stacked probability vectors từ tất cả active base models
  - VD: 3 models active → vector 6 chiều (3 models x 2 classes)
- Output: label {-1, +1}
- Meta model được train trên **OOF predictions** — không phải train predictions (tránh overfit)

### 5. Position Sizing (`src/models/main.py:HybridStackingSignalClassifier.predict_positions`)

```mermaid
flowchart TD
    A["Test probabilities<br/>từ meta-learner"] --> B{"use_meta_labeling?"}
    B -->|"True"| C["Meta-label model<br/>dự đoán P(correct)"]
    B -->|"False"| D["Confidence threshold<br/>single threshold"]
    C --> E{"P(correct) >= threshold?"}
    E -->|"Yes"| F["So sánh buy vs sell prob"]
    E -->|"No"| G["position = 0 (flat)"]
    D --> R["ADX/BB_width<br/>regime filter"]
    R --> H{"buy > sell + margin?"}
    H -->|"Yes"| I["position = +1 (buy)"]
    H -->|"No"| J{"sell > buy + margin?"}
    J -->|"Yes"| K["position = -1 (sell)"]
    J -->|"No"| L["position = 0 (flat)"]
```

**Market regime filter**: chỉ vào lệnh khi ADX >= 20.0 và BB_width >= 1.2× mean. Tránh giao dịch trong thị trường sideways/low vol.

**Chi tiết meta-labeling** (đang enabled):

```python
# Bước 1: Train meta-label model trên OOF
meta_X = np.column_stack([meta_probas, stacked])
meta_y = (primary_pred == y_enc).astype(int)  # 1 nếu meta model predict đúng

# Bước 2: Predict P(correct) trên test
P_correct = meta_label_model.predict_proba(test_meta_X)[:, 1]

# Bước 3: Chỉ vào lệnh nếu P(correct) >= threshold
if P_correct >= meta_label_threshold:
    # Chọn hướng dựa trên proba
```

## Pipeline huấn luyện đầy đủ

```mermaid
flowchart TD
    A["Train DataFrame<br/>(23,604 rows)"] --> B["Chuẩn hóa<br/>KNNImputer → StandardScaler"]
    B --> C["Label Encoding<br/>{-1, +1} → {0, 1}"]
    C --> D["PurgedEmbargoCV<br/>5 folds"]
    D --> E["OOF Training Loop"]
    E --> F["OOF Scores"]
    F --> G["Smart Filtering"]
    G --> H["Stack OOF probas"]
    H --> I["Fit Meta-Learner"]
    I --> J["Fit Active Models<br/>full train data"]
    J --> K["Fit Meta-Label Model<br/>(OOF stacking)"]
    K --> L["HybridStackingSignalClassifier<br/>sẵn sàng predict"]
```

## Kết quả OOF gần nhất

| Model | OOF macro F1 | Status |
|---|---|---|
| **GRU** | 0.413 (kiến trúc cũ) | ACTIVE |
| **LightGBM** | 0.409 (kiến trúc cũ) | ACTIVE |
| **SVC** | 0.391 (kiến trúc cũ) | ACTIVE |

Cả 3 model đều vượt ngưỡng `MIN_OOF_F1=0.36`.

## Sample Weights

```python
# src/models/stacking.py:compute_class_weights
def compute_class_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    weight_map = {c: len(y) / (len(classes) * cnt)
                  for c, cnt in zip(classes, counts)}
    return np.array([weight_map[v] for v in y])
```

- Mỗi class được weight nghịch đảo với tần suất
- Được truyền qua sklearn pipeline: `{stepname__sample_weight: weights}`

## File tham chiếu

- `src/models/main.py`: `HybridStackingSignalClassifier`
- `src/models/gru.py`: `GRUClassifier`, `GRUNet`, `FocalLoss`, `derive_rolling_sequences`
- `src/models/builders.py`: `assemble_base_model_registry`, `create_*_classifier`, `create_meta_classifier`, `create_meta_label_classifier`
- `src/models/stacking.py`: `select_qualified_oof_predictions`, `compute_class_weights`, `cross_validate_oof_probabilities`
- `src/validation/main.py`: `PurgedEmbargoTimeSeriesSplit`
- `src/config/constants.py`: `MIN_OOF_F1`, `CONFIDENCE_THRESHOLD`, `USE_META_LABELING`, `META_LABEL_THRESHOLD`, `SHORT_META_LABEL_THRESHOLD`, `ADX_THRESHOLD`, `BB_WIDTH_MIN_MULT`
