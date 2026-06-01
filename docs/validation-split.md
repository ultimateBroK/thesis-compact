# Validation & Train/Test Split

## Mục đích

Chia dữ liệu chuỗi thời gian mà không bị **data leakage** (thông tin từ tương lai rò rỉ vào quá khứ). Sử dụng hai cơ chế:

1. **Train/Test split** với **purge gap** — ngăn test labels dùng thông tin từ tương lai
2. **PurgedEmbargoTimeSeriesSplit** cho cross-validation — ngăn leakage giữa các fold

## Luồng tổng thể

```mermaid
flowchart TD
    A["Labeled DataFrame<br/>29,245 rows"] --> B["Train/Test Split<br/>80% / 20%"]
    B --> C{"Kiểm tra event_end<br/>có overlap vào test?"}
    C -->|"Có overlap<br/>event_end[train] > test_start"| D["Mở rộng purge gap"]
    C -->|"Không overlap"| E["Dùng purge mặc định<br/>2%"]
    D --> F["Purge gap = max(2%, extra)"]
    E --> F
    F --> G["Train set: 0 → split_idx<br/>Test set: split_idx + purge → end"]
    G --> H["PurgedEmbargoTimeSeriesSplit<br/>5 folds cho train"]
    H --> I["Training loop<br/>với cross-validation"]

    style A fill:#c084fc,stroke:#e9d5ff
    style G fill:#60a5fa,stroke:#93c5fd
    style H fill:#34d399,stroke:#6ee7b7
```

## 1. Train/Test Split với Purge Gap (`src/dataset/builder.py:derive_train_test_split`)

```mermaid
sequenceDiagram
    participant D as Dataset
    participant S as Splitter
    participant T as Train set
    participant Ts as Test set

    D->>S: derive_train_test_split(frame, test_size=0.2, purge_pct=0.02)
    S->>S: split = int(N * 0.8)  # 23,396 rows
    S->>S: purge = int(N * 0.02) # 585 rows
    S->>S: Kiểm tra event_end[train] có vượt test_start không?
    Note over S: Nếu event_end max > split + purge<br/>→ purge = event_end_max - split + 1
    S->>T: frame.head(split)
    S->>Ts: frame.slice(split + purge)
```

### Minh họa split

```mermaid
gantt
    title Train/Test Split Timeline
    dateFormat  YYYY-MM-DD
    axisFormat  %Y-%m

    section Dataset
    Train       :train, 2019-01-03, 2022-12-29
    Purge Gap   :purge, after train, 38d
    Test        :test, after purge, 2023-12-29
```

### Tại sao cần purge gap?

```
Train event window: |-------event_end--------|
                                Test start: |---|
```

Nếu `event_end` của một sample trong train **vượt quá** `test_start`, thì sample đó chứa thông tin giá từ tương lai (từ góc nhìn của test set). Purge gap loại bỏ vùng overlap này.

## 2. PurgedEmbargoTimeSeriesSplit (`src/validation/split.py`)

### So sánh với CV thông thường

```mermaid
flowchart LR
    subgraph "Standard K-Fold"
        direction LR
        A1["Fold 1: train"] ~~~ A2["Fold 1: val"]
        B1["Fold 2: train<br/>(bị leakage từ fold 1 val)"] ~~~ B2["Fold 2: val"]
    end

    subgraph "PurgedEmbargo CV"
        direction LR
        C1["Fold 1: train<br/>+ embargo"] ~~~ C2["Fold 1: val"]
        D1["Fold 2: train<br/>(purge + embargo)<br/>không leakage"] ~~~ D2["Fold 2: val"]
    end
```

### Cơ chế Purge + Embargo

```mermaid
flowchart TD
    A["PurgedEmbargoTimeSeriesSplit<br/>n_splits=5, embargo_pct=0.02"] --> B["Tính test_size<br/>= N // (n_splits + 1)"]
    B --> C["Với mỗi fold i:"]
    C --> D["test_start = (i+1) * test_size"]
    C --> E["test_end = test_start + test_size"]
    C --> F["embargo = ceil(N * 0.02) = ~590 rows"]
    D --> G["Bước 1: Loại bỏ sample có<br/>event_end >= test_start<br/>(purge)"]
    E --> G
    F --> H["Bước 2: Loại bỏ sample trong<br/>vùng embargo sau test_end"]
    G --> I["Bước 3: Chỉ giữ train_idx < test_start"]
    H --> I
    I --> J["yield train_idx, test_idx"]
```

### Minh họa 5 folds

```mermaid
block-beta
    columns 6
    block:train1
        columns 1
        T1["Train Fold 1"]
    end
    block:gap1
        columns 1
        G1["E"]
    end
    block:val1
        columns 1
        V1["Val Fold 1"]
    end
    space:3

    block:train2
        columns 1
        T2["Train Fold 2<br/>(purged)"]
    end
    block:gap2
        columns 1
        G2["E"]
    end
    block:val2
        columns 1
        V2["Val Fold 2"]
    end
    space:3

    block:train5
        columns 1
        T5["... Fold 5<br/>(purged)"]
    end
    block:gap5
        columns 1
        G5["E"]
    end
    block:val5
        columns 1
        V5["Val Fold 5"]
    end

    block_train1[" "]
    block_val1[" "]
```

**E** = Embargo zone (2% — loại bỏ samples ngay sau test set để tránh leakage từ tương lai)

### Chi tiết thuật toán (`src/validation/split.py:compute_embargo_clean_train_indices`)

```python
def compute_embargo_clean_train_indices(indices, event_end_pos, test_idx, embargo):
    # 1. Purge: loại samples có event_end nằm trong hoặc sau test window
    train_mask[(indices <= test_end) & (event_end_pos >= test_start)] = False

    # 2. Embargo: loại samples ngay sau test window
    train_mask[test_end + 1 : test_end + embargo + 1] = False

    return indices[train_mask]
```

## Thông số split (full dataset)

| Tham số | Giá trị |
|---|---|
| Labeled rows before purge | 29,245 |
| Kept rows (train+test) | 28,660 |
| Train rows | 23,396 (80%) |
| Test rows | 5,264 (18%) |
| Purge rows | 585 (2%) |
| Train end | 2023-01-03 15:00 UTC |
| Test start | 2023-02-08 07:00 UTC |
| Purge gap | ~35.7 ngày |
| CV folds | 5 |
| Embargo per fold | ~468 rows (2% of train) |

## Tại sao không dùng Shuffle?

Dữ liệu chuỗi thời gian tài chính có:
- **Autocorrelation**: giá hôm nay correlated với giá hôm qua
- **Look-ahead bias**: nếu shuffle, mô hình học được patterns từ tương lai
- PurgedEmbargo giải quyết cả hai vấn đề trên

## File tham chiếu

- `src/validation/split.py`: `PurgedEmbargoTimeSeriesSplit`
- `src/validation/split.py`: `compute_embargo_clean_train_indices()`
- `src/dataset/builder.py`: `derive_train_test_split()`, `compute_purge_gap()`
 - `src/config/constants.py`: `CV_SPLITS`, `EMBARGO_PCT`, `PURGE_PCT`, `TEST_SIZE`
