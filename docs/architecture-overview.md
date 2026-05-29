# Tổng quan kiến trúc Pipeline

## Mục đích

Pipeline dự báo tín hiệu giao dịch **XAU/USD CFD** (Vàng/Gold) từ dữ liệu tick-level. Sử dụng **hybrid stacking ensemble** kết hợp GRU (PyTorch), LightGBM, và SVC với purged-embargo cross-validation để tránh data leakage.

## Luồng tổng thể

```mermaid
flowchart TD
    A["Dữ liệu Tick<br/>XAU/USD Parquet"] --> B["OHLC Aggregation<br/>Polars streaming → 1h"]
    B --> C["Feature Engineering<br/>20 features"]
    C --> D["Triple-Barrier Labeling<br/>{-1, +1}"]
    D --> E["Train/Test Split<br/>80/20 + Purge Gap"]
    E --> F["Purged-Embargo CV<br/>5 folds, embargo 2%"]
    F --> G["Base Models Training<br/>GRU | LightGBM | SVC"]
    G --> H{"Smart Filtering<br/>OOF F1 >= 0.36?"}
    H -->|"Pass"| I["Active Models"]
    H -->|"Fail"| J["FILTERED<br/>(use best if all fail)"]
    I --> K["Stack OOF Proba"]
    J --> K
    K --> L["Meta-Learner<br/>LogisticRegression"]
    L --> M["Predict Test Set"]
    M --> N["Position Sizing<br/>confidence threshold 0.15"]
    N --> O["Cost-Aware Backtest<br/>Spread + Slippage"]
    O --> P["Reports & Artifacts<br/>JSON + PNG + CSV"]

    style A fill:#c084fc,stroke:#e9d5ff
    style L fill:#60a5fa,stroke:#93c5fd
    style O fill:#34d399,stroke:#6ee7b7
    style P fill:#fb923c,stroke:#fdba74
```

## Kiến trúc module

```mermaid
graph LR
    subgraph CLI
        cli["cli.py<br/>Orchestrator"]
    end

    subgraph "Data Layer"
        data["data/<br/>Parquet → OHLC"]
        dataset["dataset/<br/>Build dataset + auto-tune"]
    end

    subgraph "Feature & Label"
        features["features/<br/>21 indicators"]
        labeling["labeling/<br/>Triple barrier"]
    end

    subgraph "Model"
        validation["validation/<br/>PurgedEmbargoCV"]
        models["models/<br/>HybridStackingClassifier"]
    end

    subgraph "Evaluation"
        backtest["backtest/<br/>Equity simulation"]
        reporting["reporting/<br/>Reports + Artifacts"]
    end

    cli --> data
    data --> dataset
    dataset --> features
    dataset --> labeling
    dataset --> models
    models --> validation
    models --> backtest
    models --> reporting
    backtest --> reporting
```

## Cấu trúc thư mục

```
.
├── main.py                        # Entrypoint
├── src/
│   ├── __init__.py                # Module docstring
│   ├── cli/                       # CLI + pipeline orchestration
│   ├── config/                    # Hằng số + Pipeline/TradingCosts configs
│   ├── data/                      # Đọc parquet + resampling OHLC
│   ├── dataset/                   # Build dataset: features + labels + split + auto-tune
│   ├── features/                  # Feature engineering: 21 features
│   ├── labeling/                  # Triple-barrier labeling (swing H/L + ATR fallback)
│   ├── models/                    # GRU, LightGBM, SVC + Stacking + Meta-label
│   ├── backtest/                  # Backtest mô phỏng equity barrier-based
│   ├── reporting/                 # Báo cáo + artifacts (console + file)
│   └── validation/                # PurgedEmbargoTimeSeriesSplit
├── data/XAUUSD/                   # Dữ liệu parquet đầu vào
├── reports/run_*/                 # Artifacts đầu ra
├── docs/                          # Tài liệu
├── pixi.toml                      # Dependencies
└── viz.ipynb                      # Notebook phân tích
```

## Thông số cấu hình chính (`src/config/`)

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `TIMEFRAME` | `1h` | Khung thời gian OHLC |
| `FRACTIONAL_D` | `0.4` | Bậc fractional differencing |
| `CV_SPLITS` | `5` | Số fold cross-validation |
| `EMBARGO_PCT` | `0.02` | Tỷ lệ embargo (2%) |
| `PURGE_PCT` | `0.02` | Tỷ lệ purge gap (2%) |
| `MIN_OOF_F1` | `0.36` | Ngưỡng smart filtering |
| `CONFIDENCE_THRESHOLD` | `0.35` | Ngưỡng confidence position |
| `USE_META_LABELING` | `true` | Bật meta-labeling? |
| `META_LABEL_THRESHOLD` | `0.55` | Ngưỡng P(correct) cho long |
| `SHORT_META_LABEL_THRESHOLD` | `0.60` | Ngưỡng P(correct) cho short |
| `INITIAL_BALANCE` | `$10,000` | Vốn khởi đầu |
| `CONTRACT_SIZE` | `100` | Kích thước hợp đồng |
| `RISK_PER_TRADE` | `0.01` | 1% rủi ro mỗi lệnh (sized by stop distance) |
| `LEVERAGE` | `30` | Đòn bẩy tài khoản |
| `LABELING_HORIZON` | `24` | Vertical barrier (nến) |
| `FALLBACK_TP_ATR` | `2.0` | ATR multiplier cho TP fallback |
| `FALLBACK_SL_ATR` | `2.0` | ATR multiplier cho SL fallback |
| `MAX_LOSS_ATR` | `3.0` | Max loss barrier (ATR) |
| `AUTO_TUNE_BARRIERS` | `true` | Tự động calibrate TP/SL |
| `TUNE_TARGET_BALANCE` | `0.50` | Target class balance cho 2 classes {-1, +1} |
| `ADX_THRESHOLD` | `20.0` | Ngưỡng ADX cho regime filter |

## Kết quả tham khảo (run mẫu 2019-2023, kiến trúc cũ)

| Metric | Giá trị |
|---|---|
| Dataset | 29,505 rows (80% train / 20% test) |
| 20 features, 2 classes {-1, +1} | ~50/50 (TUNE_TARGET_BALANCE=0.50) |
| OOF F1 (GRU) | 0.413 |
| OOF F1 (LightGBM) | 0.409 |
| OOF F1 (SVC) | 0.391 |
| Test F1 macro | 0.378 |
| Total Return | -8.84% |
| Sharpe | -1.72 |
| Max DD | -11.93% |
| Training time | ~560s |

## File tham chiếu

- `src/cli/pipeline.py`: `run_pipeline()` — toàn bộ pipeline chạy tuần tự
- `src/config/constants.py` + `src/config/pipeline.py`: tất cả hằng số và dataclass config
- `src/models/main.py`: `HybridStackingSignalClassifier` — model chính
- `main.py`: `from src.cli import main`
