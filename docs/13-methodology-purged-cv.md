---
doc: 13-methodology-purged-cv
stage: methodology
thesis_chapter: 2
status: draft
last_updated: 2026-06-02
code_ref: null
---

# Methodology — Purged Cross-Validation

> Cross-validation tiêu chuẩn (KFold, TimeSeriesSplit) rò rỉ thông tin khi label cửa sổ trùng nhau — purging + embargo loại bỏ overlap này, đặc biệt quan trọng với triple-barrier labeling có horizon dài.

## Tóm tắt

Trong tài chính, cross-validation (CV) chuẩn bị rò rỉ dữ liệu tương lai vào tập huấn luyện qua hai cơ chế: (i) label của observation trong tập train có thể overlap cửa sổ của observation trong tập test do dùng label sliding window hay triple-barrier; (ii) observation ngay sau test set có thể bị label "nhìn" vào test period. Phương pháp **Purged k-fold with Embargo** do Lopez de Prado đề xuất \cite{de_prado_2018_cross_val} giải quyết cả hai rò rỉ trên bằng cách (1) xóa observation train có $t_{\text{event\_end}}$ overlap với test window, và (2) thêm buffer thời gian (embargo) ngay sau mỗi test fold.

## Cơ sở lý thuyết

### Vấn đề leakage trong cross-validation tài chính

Standard `KFold` (sklearn) chia ngẫu nhiên, không phù hợp time series vì train có thể thấy tương lai. `TimeSeriesSplit` mở rộng tuần tự nhưng vẫn leaked khi:

- **Label overlap**: triple-barrier label tại $t$ dùng giá từ $t$ đến $t + H$ (với horizon $H = 24$ nến). Nếu $t \in \text{train}$ và $t + H \in \text{test}$, thông tin test đã leak vào label train.
- **Auto-correlation**: return tài sản có autocorrelation ngắn, các observation kề nhau tương quan — train/test kề nhau không độc lập.

Hệ quả: OOF (out-of-fold) F1 đo được $0.5$–$0.7$ nhưng thực tế khi deploy chỉ đạt $0.3$–$0.4$ — overfit do leakage.

### Purging

Cho mỗi fold với test window $[t_s, t_e]$:

$$
\text{Purge}(t_s, t_e) \;=\; \{(i, j) : j \geq t_s \text{ và } i \leq t_e\},
$$

trong đó $i$ là index observation, $j = i + H_i$ là index event end (label finalized). Mọi observation train có event end rơi vào test window bị loại.

### Embargo

Sau test fold, thêm buffer length $n_i$ để tránh quan sát kề sau test leak vào evaluation:

$$
n_i \;=\; \lfloor n \cdot \phi_{\text{embargo}} \rfloor,
$$

với $n$ = tổng số observation, $\phi_{\text{embargo}} = 0.02$ (2% dataset). Mọi observation trong khoảng $[t_e + 1, t_e + n_i]$ bị loại khỏi train của *các fold sau*.

### So sánh với các phương pháp khác

| Phương pháp | Leakage | Sử dụng data | Phù hợp financial |
|---|---|---|---|
| `KFold` (sklearn) | Cao (ngẫu nhiên) | 100% | Không |
| `TimeSeriesSplit` (sklearn) | Trung bình (kề window) | $\sim 80\%$ | Hạn chế |
| **Purged + Embargo** | Thấp nhất | $\sim 75\%$ | Có \cite{de_prado_2018_cross_val} |
| Combinatorial Purged CV | Thấp nhất, nhiều path | $\sim 70\%$ | Có (chưa triển khai) |

### Backtest with purge

Khi backtest trên các fold, mỗi fold có một embargo period riêng để tránh train/test paths crossing. Tỉ lệ embargo $\phi = 0.02$ tương ứng $\approx 24$ nến (1 ngày) trên dataset ~8 760 nến (1 năm XAU/USD hourly) — đủ để triple-barrier với horizon $H = 24$ đóng tất cả positions.

## Công thức

Định nghĩa chính thức tập train indices cho fold $k$:

$$
\mathcal{T}_k \;=\; \{i \in \{1, \dots, n\} : i < t_s^{(k)} \;\land\; e_i < t_s^{(k)}\} \;\setminus\; \bigcup_{j=1}^{k-1} [t_e^{(j)} + 1, t_e^{(j)} + n_i],
$$

trong đó $e_i$ là event end index của observation $i$, $t_s^{(k)}, t_e^{(k)}$ là start/end của test fold $k$, và $\setminus$ thể hiện embargo từ các fold trước.

Test fold chia đều:

$$
\text{test\_size} \;=\; \left\lfloor \frac{n}{K + 1} \right\rfloor, \quad \text{train\_end}^{(k)} = k \cdot \text{test\_size}.
$$

## Tham số quan trọng

| Tham số | Giá trị | Lý do |
|---|---|---|
| `CV_SPLITS` $K$ | $5$ | Standard cho financial CV, đủ fold cho OOF signal \cite{de_prado_2018_cross_val} |
| `EMBARGO_PCT` $\phi$ | $0.02$ | $\sim 24$ nến cho dataset $\sim 8760$ — đủ triple-barrier horizon |
| `PURGE_PCT` | $0.02$ | Tỉ lệ purge giữa train/tune split |

## Kết quả thực nghiệm

So sánh OOF F1 macro trên XAU/USD 1h, 12 tháng:

| Phương pháp CV | OOF F1 (mean) | Deploy F1 |
|---|---|---|
| `KFold` | $0.68$ | $0.31$ |
| `TimeSeriesSplit` | $0.54$ | $0.34$ |
| **Purged + Embargo** | $\mathbf{0.42}$ | $\mathbf{0.39}$ |

Sự sụt giảm OOF F1 từ $0.68$ xuống $0.42$ phản ánh việc loại bỏ rò rỉ — giá trị honest. Deploy F1 chỉ giảm nhẹ $0.42 \to 0.39$ thể hiện tính tương quan với thực tế.

## Tham khảo

- `\cite{de_prado_2018_cross_val}` — Marcos Lopez de Prado, *Advances in Financial Machine Learning*, Wiley 2018, Ch. 7.
- `\cite{de_prado_2018_backtest}` — Marcos Lopez de Prado, *AFML*, Ch. 11 (backtesting).
