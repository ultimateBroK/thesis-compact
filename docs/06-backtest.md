---
doc: 06-backtest
stage: backtest
thesis_chapter: 3
status: draft
last_updated: 2026-06-02
code_ref: src/backtest.py
---

# Backtest

> Engine mô phỏng barrier-exit theo chuẩn CFD: bid/ask-aware barriers, half-spread round-trip cost, commission, overnight swap, lot granularity, margin guard. Module `src/backtest.py` — entrypoint `run_barrier_backtest`.

## Tóm tắt

Module `src/backtest.py` mô phỏng giao dịch XAU/USD CFD dựa trên vector position từ `HybridStackingSignalClassifier.predict_positions`. Mỗi position entry chỉ kích hoạt khi signal **thay đổi**, đóng khi một trong ba barrier bị chạm: take-profit (TP), stop-loss (SL), hoặc signal reversal. Position sizing theo nguyên tắc risk-based: 2% equity rủi ro mỗi trade, lot tính từ stop distance và làm tròn theo `LOT_STEP` (0.01). Cost model đầy đủ CFD: half-spread tại entry + half-spread tại exit, commission hai chiều, overnight swap theo direction. Engine guard margin theo `LEVERAGE` — skip trade nếu margin yêu cầu vượt equity khả dụng. Engine output gồm dict metrics, list trade records, và equity array. Hàm `search_backtest_parameters` grid-search tham số `(min_hold, tp_atr, sl_atr)` tối ưu Sharpe trên train data.

## Cơ sở lý thuyết

Mô phỏng dựa trên barrier exit \cite{de_prado_2018_afml}: mỗi trade có hai horizontal barrier (TP/SL) đặc tả mức giá thoát; khi giá chạm barrier đầu tiên, trade đóng với PnL tương ứng. Position sizing theo risk fraction \cite{vince_2009_optimal_f}: số lot $L$ sao cho nếu SL bị chạm, loss $\leq$ `RISK_PER_TRADE` $\cdot$ equity.

Mở rộng so với engine đơn giản: mô hình hoá **bid/ask spread** đúng cách. Trong CFD broker, trader mua tại ask (mid $+$ half-spread) và bán tại bid (mid $-$ half-spread). Một round-trip (mở + đóng) tốn đúng **một spread đầy đủ**, chia đều entry/exit. Engine cũ tính `spread × lots × cs × 2` (double-count), engine mới sửa thành `(s_entry + s_exit) / 2 × lots × cs`. Ngoài ra TP/SL trigger cũng được điều chỉnh: TP long kích hoạt khi bid-chạm TP (mid $\geq$ TP $+$ half-spread), SL long kích hoạt khi bid-chạm SL — phản ánh thực tế fill tại bid khi close position.

## Công thức

### Entry & barriers

Tại bar $i$, nếu signal thay đổi và $\text{pos}_i \neq 0$:

$$
P_{\text{entry}} = C_i, \qquad \text{ATR}_i^{\$} = \text{ATR}^{rel}_i \cdot C_i,
$$

$$
P_{\text{TP}} = P_{\text{entry}} + d_{\text{TP}} \cdot \text{ATR}_i^{\$} \cdot \text{sgn}(\text{pos}_i), \qquad P_{\text{SL}} = P_{\text{entry}} - d_{\text{SL}} \cdot \text{ATR}_i^{\$} \cdot \text{sgn}(\text{pos}_i),
$$

với $d_{\text{TP}}$ được chọn bởi `search_backtest_parameters` trong range `TUNE_TP_RANGE_BT`, $d_{\text{SL}}$ từ `TUNE_SL_RANGE_BT`. Barriers tính trong hệ toạ độ mid-price.

### Barrier triggers (bid/ask-aware)

Tại exit bar $j$, half-spread $h_j = s_j / 2$:

| Direction | Trigger TP | Trigger SL |
|---|---|---|
| LONG  | $H_j \geq P_{\text{TP}} + h_j$ (bid reaches TP) | $L_j \leq P_{\text{SL}} + h_j$ (bid reaches SL) |
| SHORT | $L_j \leq P_{\text{TP}} - h_j$ (ask reaches TP) | $H_j \geq P_{\text{SL}} - h_j$ (ask reaches SL) |

Fill price = barrier level (TP hoặc SL). Half-spread adjustment chỉ tác động lên điều kiện trigger, không thay đổi fill price — chi phí spread được khấu trừ riêng ở phần cost.

### Position sizing

Stop distance $\Delta_s = |P_{\text{entry}} - P_{\text{SL}}|$, risk budget $B = E_i \cdot r$ với $r=$ `RISK_PER_TRADE`:

$$
L_{\text{raw}} = \frac{B}{\Delta_s \cdot K}, \qquad L = \text{round\_step}(L_{\text{raw}}, \text{LOT\_STEP}), \quad L \in [\text{LOT\_MIN}, \text{LOT\_MAX}],
$$

với $K = $ `CONTRACT_SIZE` = 100 (1 lot XAU = 100 oz). `round_step` snap về bội số gần nhất của `LOT_STEP` (mặc định 0.01) — mô phỏng broker lot granularity. Hard bounds `[LOT_MIN=0.01, LOT_MAX=5.0]`. SHORT positions scaled by `SHORT_LOT_SCALE=0.20` sau `round_lot`, trước margin check — giảm SHORT exposure do higher asymmetric risk.

### Margin guard

Margin yêu cầu cho position:

$$
M = \frac{L \cdot K \cdot P_{\text{entry}}}{\text{LEVERAGE}}.
$$

Nếu $M > E_i$ thì trade bị **skip** (không đủ margin theo leverage broker). Mặc định `LEVERAGE = 100` (1% margin).

### PnL & cost

Gross PnL thoát tại giá $P_{\text{exit}}$:

$$
\text{PnL}_{\text{gross}} = \text{sgn}(\text{pos}) \cdot (P_{\text{exit}} - P_{\text{entry}}) \cdot L \cdot K.
$$

Spread cost (round-trip, half-spread mỗi side):

$$
C_{\text{spread}} = \frac{s_{\text{entry}} + s_{\text{exit}}}{2} \cdot L \cdot K.
$$

Commission (cả hai chiều):

$$
C_{\text{comm}} = 2 \cdot c_{\text{lot}} \cdot L,
$$

với $c_{\text{lot}} =$ `COMMISSION_PER_LOT_SIDE`. Default 0 (spread-only broker).

Overnight swap:

$$
C_{\text{swap}} = N_{\text{overnights}} \cdot r_{\text{swap}}(\text{direction}) \cdot L,
$$

với $N_{\text{overnights}}$ = số ngày UTC crossed giữa entry và exit, $r_{\text{swap}}$ = `SWAP_LONG_USD_PER_LOT` (long) hoặc `SWAP_SHORT_USD_PER_LOT` (short). Mặc định $-2.50$ (long) và $-1.00$ (short) USD/lot/night — mức tiêu biểu XAU/USD retail broker.

Net trade PnL:

$$
\text{PnL}_{\text{net}} = \text{PnL}_{\text{gross}} - C_{\text{spread}} - C_{\text{comm}} - C_{\text{swap}}.
$$

### Equity & metrics

Equity per bar (chỉ cập nhật khi trade đóng):

$$
E_0 = E_{\text{init}}, \qquad E_i = E_{i-1} + \text{PnL}_{\text{net},i}^{\text{realized}}.
$$

Sharpe annualized (giả định 252 ngày $\times$ 24 giờ):

$$
\text{Sharpe} = \sqrt{252 \cdot 24} \cdot \frac{\mu(r)}{\sigma(r)}, \qquad r_i = \frac{E_i - E_{i-1}}{E_{i-1}}.
$$

Max drawdown: $\text{MDD} = \min_t (E_t - \max_{s \leq t} E_s) / \max_{s \leq t} E_s$. Profit factor $= \sum \text{PnL}^+ / |\sum \text{PnL}^-|$.

## Cài đặt

### Engine chính

```
run_barrier_backtest(frame, positions, tp_atr, sl_atr, initial_balance)
├── Extract close, high, low, atr_14, spread, timestamp → numpy
├── equity = full(n, initial_balance)
├── FOR i = 1 .. n-1:
│   ├── equity[i] = equity[i-1]                            # carry forward
│   ├── IF in_trade:
│   │   ├── half_spread = spread[i] / 2
│   │   ├── check barrier (bid-aware LONG / ask-aware SHORT)
│   │   ├── IF pos[i] != direction: exit_price = close[i]  # signal reversal
│   │   ├── IF exit_price:
│   │   │   ├── gross = dir * (exit - entry) * lots * contract_size
│   │   │   ├── overnights = count_utc_days(entry_idx, i)
│   │   │   ├── (sp_cost, comm, swap) = compute_trade_costs(...)
│   │   │   ├── net = gross - sp_cost - comm - swap
│   │   │   ├── append create_trade_record(...)
│   │   │   └── equity[i] += net                           # realized
│   └── IF not in_trade AND pos[i] != 0 AND pos[i] != pos[i-1]:
│       ├── compute tp_price, sl_price from ATR
│       ├── lots = round_lot(risk_budget / (stop_dist * contract_size))
│       ├── IF margin_required(lots, entry_price) > equity[i]: SKIP
│       └── in_trade = True, direction = pos[i], entry_idx = i
├── Force-close open trade at close[-1] (force exit)
├── assert |equity[-1] - (E_init + Σ trade_pnl_net)| < 1e-6  # invariant
└── return compute_backtest_metrics(...), trades, equity
```

Quy ước: khi signal thay đổi nhưng trade đang mở $\to$ exit tại `close[i]`. Entry chỉ kích hoạt khi signal đổi từ 0 hoặc chiều ngược $\to$ chiều mới — một block signal liên tục sinh tối đa 1 trade.

### Trade record

`create_trade_record(entry_idx, exit_idx, direction, entry_price, exit_price, gross_pnl, spread_cost, commission, swap, lots, overnights)`:

| Field | Dtype | Mô tả |
|---|---|---|
| `entry_idx` / `exit_idx` | int | Index trong frame |
| `direction` | str | `"LONG"` / `"SHORT"` |
| `entry_price` / `exit_price` | float | Mid price tại entry/exit |
| `lots` | float | Lot thực tế (sau round + clamp) |
| `bars_held` | int | `exit_idx - entry_idx + 1` |
| `overnights` | int | Số overnight UTC crossed |
| `gross_pnl_usd` | float | PnL trước cost |
| `spread_cost_usd` | float | Half-spread entry + half-spread exit |
| `commission_usd` | float | Commission hai chiều |
| `swap_usd` | float | Overnight swap |
| `trade_pnl_usd` | float | Net = gross $-$ spread $-$ commission $-$ swap |
| `cost_usd` | float | Tổng cost = spread + commission + swap |
| `win` | bool | `trade_pnl_usd > 0` |

### Min position hold

Backtest engine **không** tự enforce min_hold. Tham số `min_hold` được chọn bởi grid search `TUNE_HOLD_VALUES` và apply trước backtest trong `src/models.py::enforce_minimum_position_hold`. Xem `docs/05-models-stacking.md` section "Position strategy".

### Backtest tuning (always-on)

`search_backtest_parameters(model, train_data, features, close_prices, tp_range, sl_range, min_hold_values)`:

1. Predict raw positions trên train (skip_min_hold=True).
2. Grid loop qua `TUNE_TP_RANGE_BT × TUNE_SL_RANGE_BT × TUNE_HOLD_VALUES`, skip `sl >= tp`.
3. Với mỗi combo: `enforce_minimum_position_hold`, `run_barrier_backtest`, lấy `metrics["sharpe"]`.
4. Track combo tối đa Sharpe, trả về dict `{tp, sl, min_hold, score, trades, win_rate, profit_factor}`.

Tuning luôn chạy — không có fallback hardcoded. Single source of truth là `TUNE_*_RANGE_BT` và `TUNE_HOLD_VALUES` trong `src/config.py`.

### Inputs / Outputs

| Input | Source | Description |
|---|---|---|
| `frame` | `test` (Polars) | Cột `close, high, low, atr_14, spread, timestamp` |
| `positions` | `model.predict_positions(...)` | Vector $\in \{-1, 0, +1\}$ |
| `tp_atr, sl_atr` | `search_backtest_parameters` best | Barrier width |
| `initial_balance` | `INITIAL_BALANCE = 10 000` | USD |

Module constants (CFD execution model, xem `src/config.py`):

| Constant | Giá trị | Mục đích |
|---|---|---|
| `CONTRACT_SIZE` | 100 | 1 lot XAU = 100 oz |
| `RISK_PER_TRADE` | 0.02 | 2% equity risk |
| `LEVERAGE` | 100 | 1:100 → 1% margin |
| `LOT_STEP` / `LOT_MIN` / `LOT_MAX` | 0.01 / 0.01 / 5.0 | Lot granularity |
| `COMMISSION_PER_LOT_SIDE` | 0.0 | USD/lot/side (0 = spread-only broker) |
| `SWAP_LONG_USD_PER_LOT` | $-2.50$ | USD/lot/overnight (long) |
| `SWAP_SHORT_USD_PER_LOT` | $-1.00$ | USD/lot/overnight (short) |
| `SHORT_LOT_SCALE` | $0.20$ | SHORT position scale factor (sau round_lot, trước margin check) |
| `N_TUNING_TRIALS_APPROX` | $700$ | Approximate number of grid combinations in backtest tuning |

| Output | Type | Description |
|---|---|---|
| `metrics` | `dict[str, float]` | `total_return, sharpe, sortino, dsr_statistic, dsr_p_value, max_drawdown, profit_factor, win_rate, trades, trade_signals, avg_cost_usd, avg_swap_usd` |
| `trades` | `list[dict]` | Mỗi dict là một trade record |
| `equity` | `np.ndarray` | Equity curve length $n$ |

### Code refs

`src/backtest.py::create_trade_record`, `src/backtest.py::round_lot`, `src/backtest.py::compute_lots_by_risk`, `src/backtest.py::margin_required`, `src/backtest.py::compute_overnights`, `src/backtest.py::compute_trade_costs`, `src/backtest.py::run_barrier_backtest`, `src/backtest.py::compute_sharpe_ratio`, `src/backtest.py::compute_max_drawdown`, `src/backtest.py::compute_profit_factor`, `src/backtest.py::compute_win_rate`, `src/backtest.py::compute_backtest_metrics`, `src/backtest.py::search_backtest_parameters`, `src/backtest.py::compute_sortino_ratio`, `src/backtest.py::compute_deflated_sharpe_ratio`.

## Tham số quan trọng

| Tham số | Giá trị | Vị trí | Lý do |
|---|---|---|---|
| `INITIAL_BALANCE` | $\$10\,000$ | `src/config.py:29` | Vốn khởi điểm retail XAU/USD |
| `CONTRACT_SIZE` | $100$ | `src/config.py:30` | 1 lot XAU = 100 oz |
| `RISK_PER_TRADE` | $0.02$ | `src/config.py` | 2% equity risk — Kelly conservative |
| `LEVERAGE` | $100$ | `src/config.py` | 1:100 — retail broker tiêu biểu |
| `LOT_STEP` | $0.01$ | `src/config.py` | Broker minimum increment |
| `LOT_MIN` / `LOT_MAX` | $0.01$ / $5.0$ | `src/config.py` | Lot bounds |
| `COMMISSION_PER_LOT_SIDE` | $\$0$ | `src/config.py` | Spread-only broker |
| `SWAP_LONG_USD_PER_LOT` | $-\$2.50$ | `src/config.py` | Swap long tiêu biểu |
| `SWAP_SHORT_USD_PER_LOT` | $-\$1.00$ | `src/config.py` | Swap short tiêu biểu |
| `TUNE_TP_RANGE_BT` | $(3.0, 15.0, 1.0)$ | `src/config.py` | Grid search TP — single source of truth |
| `TUNE_SL_RANGE_BT` | $(3.0, 15.0, 1.0)$ | `src/config.py` | Grid search SL |
| `TUNE_HOLD_VALUES` | $[6, 8, 12, 16]$ | `src/config.py` | Grid search min_hold |

## Kết quả thực nghiệm

Smoke test 3 tháng (Q1 2019), 3 trades executed:

| Trade | Dir | Lots | Entry | Exit | Gross | Spread | Net |
|---|---|---|---|---|---|---|---|
| 1 | LONG | 0.29 | 1313.06 | 1309.66 (SL) | $-\$98.77$ | $\$7.94$ | $-\$106.71$ |
| 2 | SHORT | 0.24 | 1307.96 | 1312.04 (SL) | $-\$98.02$ | $\$6.86$ | $-\$104.88$ |
| 3 | SHORT | 0.21 | 1292.00 | 1291.13 (TP) | $+\$17.96$ | $\$6.64$ | $+\$11.32$ |

Spread cost $\approx \$7$ / trade — khớp công thức $\frac{s_{\text{entry}} + s_{\text{exit}}}{2} \cdot L \cdot 100$ với $s \approx \$0.5$ và $L \approx 0.24$ lot. Swap = 0 vì tất cả trade đóng nội ngày. Invariant: equity cuối = balance đầu + Σ net PnL (drift $< 10^{-6}$).

## Tham khảo

- `\cite{de_prado_2018_afml}` — López de Prado, *Advances in Financial Machine Learning*, Wiley 2018, Ch. 11 (backtesting risk) và Ch. 3 (triple barrier).
- `\cite{vince_2009_optimal_f}` — Vince, *The Handbook of Portfolio Mathematics*, Wiley 2009.
- `docs/03-labeling-triple-barrier.md` — barrier concept cho labeling.
- `docs/05-models-stacking.md` — `predict_positions`, `enforce_minimum_position_hold`.
- `docs/08-config.md` — bảng đầy đủ tham số.
- `docs/22-evaluation-metrics.md` — định nghĩa metrics đầy đủ.
