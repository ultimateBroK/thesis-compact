# Feature Engineering — 21 Features

## Mục đích

Xây dựng 21 đặc trưng từ dữ liệu nến OHLC 1h. Bao gồm: returns, trend indicators, momentum, volatility, volume (OBV), calendar features, signal processing (fractional differencing), và volume/spread normalization.

## Luồng xử lý

```mermaid
flowchart TD
    A["OHLC 1h DataFrame"] --> B["generate_returns_features()<br/>return_4, return_12"]
    A --> C["generate_trend_features()<br/>ema_12, ema_26, macd"]
    A --> D["generate_momentum_features()<br/>rsi_14, adx_14"]
    A --> E["generate_volatility_features()<br/>atr_14, bb_width, bb_position,<br/>volatility_24, vol_ratio_6_24,<br/>spread_z_24, close_in_range_24"]
    A --> F["generate_calendar_features()<br/>hour, dayofweek"]
    A --> G["generate_volume_features()<br/>obv, obv_delta_12"]
    A --> H["Fractional Diff<br/>close_fracdiff (d=0.4, NaN forward-filled)"]

    B --> I["Pipeline: combine_market_features()"]
    C --> I
    D --> I
    E --> I
    F --> I
    G --> I
    I --> J["enrich_with_technical_features()"]
    H --> J
    J --> K["DataFrame với 21 features +<br/>open, high, low, close, timestamp"]

    style A fill:#c084fc,stroke:#e9d5ff
    style J fill:#60a5fa,stroke:#93c5fd
    style K fill:#34d399,stroke:#6ee7b7
```

## Danh sách 21 features

```mermaid
mindmap
  root((21 Features))
    Returns
      return_4
      return_12
    Trend
      ema_12
      ema_26
      macd
    Momentum
      rsi_14
      adx_14
    Volatility
      atr_14
      bb_width
      bb_position
      volatility_24
      vol_ratio_6_24
    Spread
      spread_z_24
    Price Structure
      close_in_range_24
    Volume (OBV)
      obv
      obv_delta_12
    Calendar
      hour
      dayofweek
    Signal Processing
      close_fracdiff
    Raw
      volume
      spread
```

## Chi tiết từng nhóm

### 1. Returns (`src/features/builders.py:generate_returns_features`)

```python
close / close.shift(4) - 1   # return_4: lợi nhuận 4 nến (~4h)
close / close.shift(12) - 1  # return_12: lợi nhuận 12 nến (~12h)
```

### 2. Trend Indicators (`src/features/builders.py:generate_trend_features`)

| Feature | Công thức | Mục đích |
|---|---|---|
| `ema_12` | `EMA(close, 12) / close - 1` | Xu hướng ngắn hạn |
| `ema_26` | `EMA(close, 26) / close - 1` | Xu hướng trung hạn |
| `macd` | `EMA(close, 12) - EMA(close, 26)` | MACD line |

### 3. Momentum Indicators (`src/features/builders.py:generate_momentum_features`)

| Feature | Công thức | Mục đích |
|---|---|---|
| `rsi_14` | `100 - 100 / (1 + avg_gain / avg_loss)` | Relative Strength Index (14) |
| `adx_14` | `ADX(high, low, close, 14)` | Average Directional Index — regime filter |

### 4. Volatility Indicators (`src/features/builders.py:generate_volatility_features`)

| Feature | Công thức | Mục đích |
|---|---|---|
| `atr_14` | `ATR(high, low, close, 14) / close` | Average True Range (normalized) |
| `bb_width` | `4 * std(close, 20) / SMA(close, 20)` | Bollinger Band width |
| `bb_position` | `(close - BB_mid) / (2 * BB_std)` | Position trong band |
| `volatility_24` | `std(return_1, 24)` | Volatility 24 nến |
| `vol_ratio_6_24` | `std(return_1, 6) / std(return_1, 24)` | Tỷ lệ vol ngắn/trung — phát hiện regime change |
| `spread_z_24` | `(spread - SMA(spread, 24)) / std(spread, 24)` | Z-score của spread — phát hiện bất thường thanh khoản |
| `close_in_range_24` | `(close - low_24) / (high_24 - low_24)` | Vị trí đóng cửa trong biên độ 24 nến |

### 5. Calendar Features (`src/features/builders.py:generate_calendar_features`)

| Feature | Giá trị | Mục đích |
|---|---|---|
| `hour` | 0–23 | Giờ UTC |
| `dayofweek` | 0–6 | Effects cuối tuần |

### 6. Volume (OBV) Features (`src/features/builders.py:generate_volume_features`)

| Feature | Công thức | Mục đích |
|---|---|---|
| `obv` | `cumsum(sign(Δclose) * volume)` | On-Balance Volume — xác nhận xu hướng |
| `obv_delta_12` | `obv - obv.shift(12)` | OBV thay đổi trong 12 nến |

## Xử lý tín hiệu

### Fractional Differencing (`src/features/fractional.py:derive_fractionally_differentiated_series`)

```mermaid
flowchart LR
    A["close"] --> B["Tính weights<br/>d=0.4, threshold=1e-4"]
    B --> C["Convolution<br/>Numba JIT @njit"]
    C --> D["Forward-fill NaN<br/>281 leading NaN from window"]
    D --> E["close_fracdiff"]

    style A fill:#c084fc,stroke:#e9d5ff
    style E fill:#34d399,stroke:#6ee7b7
```

- **Mục đích**: Giữ long memory (tính dừng yếu) của chuỗi giá mà không làm mất hoàn toàn thông tin như difference bậc 1
- `d=0.4`: fractional differencing order — cân bằng giữa stationarity và memory
- `threshold=1e-4`: cắt weights nhỏ để giới hạn độ dài convolution (~282 weights, ~12 ngày lookback)
- NaN được forward-fill, 281 leading NaN đầu bị drop bởi `clean_labeled_frame`
- Dùng **Numba `@njit`** cho tốc độ

### So sánh: Fractional vs Integer Difference

```mermaid
block-beta
    columns 3
    block:g1
        columns 1
        A["Giá gốc (I(1))"] B["d=1.0 (sai phân bậc 1)"]
    end
    space
    block:g2
        columns 1
        C["Giá gốc (I(1))"] D["d=0.4 (fractional diff)"]
    end
    space
    block:g3
        columns 1
        E["Giá gốc"] F["d=0.0 (giữ nguyên)"]
    end

    g1 --> G["Mất memory<br/>Nhưng stationarity"]
    g2 --> H["Giữ memory<br/>Gần stationarity"]
    g3 --> I["Giữ memory<br/>Không stationarity"]
```

## File tham chiếu

- `src/features/builders.py`: toàn bộ feature engineering (generate_*, combine_market_features)
- `src/features/fractional.py`: `derive_fractionally_differentiated_series()`
- `src/features/builders.py`: `enrich_with_technical_features()` — aggregating call
- `src/dataset/builder.py`: `load_featured_candles()` gọi `enrich_with_technical_features()`
- `src/config/constants.py`: `FRACTIONAL_D`
