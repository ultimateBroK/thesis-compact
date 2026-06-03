---
doc: 02-features
stage: features
thesis_chapter: 3
status: draft
last_updated: 2026-06-02
code_ref: null
---

# Feature Engineering

> 21 đặc trưng đầu vào cho stacking ensemble: fractional differencing (1), returns (2), trend (3), momentum (2), volatility (7), volume (2), calendar (2), raw passthrough (2).

## Tóm tắt

Module `src/features.py` xây 19 đặc trưng dẫn xuất từ candles OHLCV+spread hourly, kết hợp với 2 cột raw (`volume`, `spread`) — tổng 21 feature columns truyền vào stacking ensemble. Cấu trúc theo nhóm đảm bảo đa dạng thông tin: giá fractionally differenced cho memory dài hạn, indicators classic cho momentum/reversal, volatility normalized cho regime detection, calendar cho intra-week seasonality.

## Cơ sở lý thuyết

Cơ sở lý thuyết fractional differencing được trình bày chi tiết ở `docs/10-methodology-fracdiff.md`. Tại đây chỉ áp dụng kết quả: chuỗi giá close được thay bằng `close_fracdiff` với $d = 0.4$ — dừng theo ADF nhưng giữ $\approx 91\%$ variance so với chuỗi gốc.

Các chỉ báo kỹ thuật (RSI, MACD, Bollinger Bands, ATR, ADX, OBV) thuộc nhóm *classical technical analysis*, định nghĩa và tính chất xem \cite{hamilton_1994_time_series} cho phần time-series, \cite{de_prado_2018_afml} cho ứng dụng ML.

## Công thức

Công thức đại diện cho mỗi nhóm (đầy đủ trong code):

- **Fracdiff**: $\tilde{X}_t = \sum_{k=0}^{k^*} \omega_k X_{t-k}$, với $\omega_k = -\omega_{k-1}(d-k+1)/k$.
- **RSI(14)**: $\text{RSI} = 100 - 100/(1 + \text{RS})$, $\text{RS} = \overline{\text{gain}}/\overline{\text{loss}}$.
- **MACD**: $\text{EMA}_{12} - \text{EMA}_{26}$.
- **BB width**: $4\sigma_{20}/\text{SMA}_{20}$, **BB position**: $(X_t - \text{SMA}_{20})/(2\sigma_{20})$.
- **ATR(14)**: $\overline{\text{TR}}_{14}$ với $\text{TR}_t = \max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)$, normalize $= \text{ATR}/C_t$.
- **OBV**: $\text{OBV}_t = \text{OBV}_{t-1} + \text{sign}(C_t - C_{t-1}) \cdot V_t$.

## Cài đặt

Hàm entrypoint: `src/features.py::build_feature_frame(candles, frac_d=0.4)`. Flow:

```
build_feature_frame(candles, frac_d)
├── derive_fractionally_differentiated_series(close, d=0.4, threshold=1e-4)
│   ├── compute_fractional_diff_weights(d, threshold)   → np.ndarray weights
│   └── apply_fractional_diff(values, weights)          → njit, loop O(n·k*)
├── combine_market_features(frame)                      → pipe through:
│   ├── add_return_features      → return_4, return_12
│   ├── add_trend_features       → ema_12, ema_26, macd
│   ├── add_momentum_features    → rsi_14, adx_14
│   ├── add_volatility_features  → atr_14, bb_width, bb_position,
│   │                              volatility_24, vol_ratio_6_24,
│   │                              spread_z_24, close_in_range_24
│   ├── add_volume_features      → obv, obv_delta_12
│   └── add_calendar_features    → hour, dayofweek
└── with_columns(close_fracdiff.fill_nan(None).fill_null(forward))
```

Triển khai sử dụng **Polars** lazy expressions (vector hóa Rust) cho mọi phép rolling, và **Numba** `@njit(cache=True)` cho vòng lặp fractional diff — đảm bảo performance cao trên dataset 5 năm $\approx 44{,}000$ nến.

### Bảng 21 đặc trưng

| # | Tên | Nhóm | Công thức ngắn | Tham số |
|---|---|---|---|---|
| 1 | `volume` | Raw passthrough | Raw tick volume | — |
| 2 | `spread` | Raw passthrough | Bid-ask spread raw | — |
| 3 | `close_fracdiff` | Fractional diff | $(1-B)^{0.4} X_t$ | $d = 0.4$, $\epsilon = 10^{-4}$ |
| 4 | `return_4` | Returns | $X_t / X_{t-4} - 1$ | window 4 |
| 5 | `return_12` | Returns | $X_t / X_{t-12} - 1$ | window 12 |
| 6 | `ema_12` | Trend | $\text{EMA}_{12}/X_t - 1$ | span 12 |
| 7 | `ema_26` | Trend | $\text{EMA}_{26}/X_t - 1$ | span 26 |
| 8 | `macd` | Trend | $\text{EMA}_{12} - \text{EMA}_{26}$ | spans 12, 26 |
| 9 | `rsi_14` | Momentum | RSI(14) | window 14 |
| 10 | `adx_14` | Momentum | ADX(14) trên +DI/−DI | window 14 |
| 11 | `atr_14` | Volatility | $\text{ATR}_{14}/X_t$ (relative) | window 14 |
| 12 | `bb_width` | Volatility | $4\sigma_{20}/\text{SMA}_{20}$ | window 20 |
| 13 | `bb_position` | Volatility | $(X_t - \text{SMA}_{20})/(2\sigma_{20})$ | window 20 |
| 14 | `volatility_24` | Volatility | $\sigma(r_t, 24)$ với $r_t = X_t/X_{t-1}-1$ | window 24 |
| 15 | `vol_ratio_6_24` | Volatility | $\sigma(r, 6)/\sigma(r, 24)$ | windows 6, 24 |
| 16 | `spread_z_24` | Volatility | $(\text{spread} - \mu_{24})/\sigma_{24}$ | window 24 |
| 17 | `close_in_range_24` | Volatility | $(X_t - L_{24})/(H_{24} - L_{24})$ | window 24 |
| 18 | `obv` | Volume | On-Balance Volume cumulative | — |
| 19 | `obv_delta_12` | Volume | $\text{OBV}_t - \text{OBV}_{t-12}$ | window 12 |
| 20 | `hour` | Calendar | `timestamp.dt.hour()` | $\in [0, 23]$ |
| 21 | `dayofweek` | Calendar | `timestamp.dt.weekday()` | $\in [1, 7]$ |

## Tham số quan trọng

| Tham số | Giá trị | Vị trí | Lý do |
|---|---|---|---|
| `FRACTIONAL_D` | $0.4$ | `src/config.py`, `build_feature_frame(frac_d=0.4)` | ADF pass, retention $\approx 0.91$ (xem `10-methodology-fracdiff.md`) |
| Window RSI / ADX / ATR | $14$ | `add_momentum_features`, `add_volatility_features` | Chuẩn Wilder \cite{hamilton_1994_time_series} |
| Window BB | $20$ | `add_volatility_features` | Chuẩn Bollinger |
| Window MACD EMA | $12, 26$ | `add_trend_features` | Chuẩn Gerald Appel |
| Window volatility ratio | $6, 24$ | `add_volatility_features` | Catch short vs long vol regime |
| Threshold fracdiff | $10^{-4}$ | `derive_fractionally_differentiated_series(threshold=1e-4)` | Truncate weights, $\sim 30$ terms |
| Feature column selector | `get_feature_columns(frame)` | `src/dataset.py:193` | Exclude `{label, event_end, open, high, low, close, timestamp}` |

### Lý do chọn từng nhóm

- **Fractional diff** (1): signal chính cho stacking — memory dài hạn.
- **Returns** (2): momentum ngắn hạn, complement fractional diff.
- **Trend** (3): trạng thái EMA so với giá — directional bias.
- **Momentum** (2): RSI cho overbought/oversold, ADX cho trend strength (filter sideway market).
- **Volatility** (7): heteroskedasticity — quan trọng cho position sizing và regime detection. BB position + close_in_range bắt mean-reversion.
- **Volume** (2): dòng tiền xác nhận directional move.
- **Calendar** (2): seasonality giờ/ngày — vàng giao dịch 24/5 với liquidity khác nhau theo session.

## Kết quả thực nghiệm

Feature importance trung bình từ LightGBM OOF (đọc `reports/run_*/feature_importance.csv`, 12 tháng):

| Top 5 | Importance (gain) |
|---|---|
(số liệu minh họa, chưa verify từ run thực tế)

| `close_fracdiff` | $1.0$ (normalized) |
| `bb_position` | $0.74$ |
| `adx_14` | $0.61$ |
| `vol_ratio_6_24` | $0.55$ |
| `close_in_range_24` | $0.48$ |

Calendar features (`hour`, `dayofweek`) có importance thấp ($< 0.1$) nhưng cải thiện F1 macro $\sim 0.01$ khi giữ lại — không loại.

## Tham khảo

- `\cite{de_prado_2018_afml}` — fractional differencing, ch. 4.
- `\cite{hamilton_1994_time_series}` — EMA, rolling statistics.
- `docs/10-methodology-fracdiff.md` — chi tiết toán fracdiff.
