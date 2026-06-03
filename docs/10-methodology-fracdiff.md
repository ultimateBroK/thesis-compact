---
doc: 10-methodology-fracdiff
stage: methodology
thesis_chapter: 2
status: draft
last_updated: 2026-06-02
code_ref: null
---

# Methodology — Fractional Differencing

> Bài toán giữ memory: phân tích chuỗi tài sản thường yêu cầu dừng (stationarity), nhưng vi phân nguyên (integer differencing) xóa sạch ký ức dài hạn — fractional differencing cho $d \in (0,1)$ bảo toàn một phần memory mà vẫn đạt dừng.

## Tóm tắt

Fractional differencing mở rộng toán tử vi phân $B$ từ bậc nguyên sang bậc thực $d \in (0,1)$, sinh ra một chuỗi vừa dừng vừa còn thông tin dài hạn cần thiết cho dự báo. Luận văn chọn $d = 0.4$ dựa trên hai tiêu chí: kiểm định Dickey-Fuller tăng (ADF) chấp nhận giả thuyết dừng ở mức ý nghĩa $5\%$, và tỉ lệ phương sai (variance retention) so với chuỗi gốc vượt $0.9$.

## Cơ sở lý thuyết

### Bài toán trade-off

Mô hình học máy giả định đầu vào dừng (stationarity). Với chuỗi giá tài sản, thông lệ là lấy sai phân bậc một $X_t - X_{t-1}$ — kết quả chuỗi dừng nhưng triệt tiêu *memory*: mọi cấu trúc dài hạn (chu kỳ, momentum, mean-reversion) bị mất. Khi dự báo tín hiệu giao dịch, cả hai thành phần đều quan trọng: dừng để model hội tụ, memory để có tín hiệu dự báo \cite{de_prado_2018_afml}.

### Định nghĩa toán operator

Toán tử trễ $B$ định nghĩa $B X_t = X_{t-1}$. Chuỗi thương của $B$ với bậc nguyên $d$:

$$
(1 - B)^d X_t \;=\; \sum_{k=0}^{d} (-1)^k \binom{d}{k} X_{t-k}.
$$

Tổng quát cho $d \in (0,1)$ sử dụng khai triển Newton cho $(1 - B)^d$ với lũy thừa thực:

$$
\boxed{\;\tilde{X}_t \;=\; (1 - B)^d X_t \;=\; \sum_{k=0}^{\infty} \omega_k \, X_{t-k}, \quad \omega_k = (-1)^k \binom{d}{k}\;}
$$

với hệ số nhị thức tổng quát

$$
\omega_k \;=\; \prod_{i=1}^{k} \frac{i - 1 - d}{i} \;=\; \omega_{k-1} \cdot \left(1 - \frac{d}{k}\right).
$$

### Decay của hệ số

Tỉ lệ hai số hạng liên tiếp

$$
\frac{\omega_k}{\omega_{k-1}} \;=\; 1 - \frac{d}{k},
$$

tiến về $1$ khi $k \to \infty$. Nghĩa là $|\omega_k| \to 0$ nhưng **không bao giờ triệt tiêu hoàn toàn** — chuỗi có memory vô hạn về lý thuyết.

### Cắt ngắn thực hành

Truncate tại ngưỡng $\epsilon$ để có hữu hạn hệ số:

$$
k^* \;=\; \min \{k : |\omega_k| < \epsilon\}.
$$

Với $\epsilon = 10^{-4}$ (giá trị mặc định trong `src/features.py::compute_fractional_diff_weights`), số hệ số hữu hạn thường nằm trong khoảng $20$–$50$ cho $d = 0.4$.

### Lower bound để memory vô hạn

Variance retention sau khi vi phân bậc $d$:

$$
\mathrm{Var}(\tilde{X}) \;\approx\; \mathrm{Var}(X) \cdot \sum_{k=0}^{\infty} \omega_k^2.
$$

Tổng $\sum \omega_k^2$ giảm khi $d$ tăng. Giới hạn dưới $d < \epsilon_d$ (với $\epsilon_d$ nhỏ, thường $\approx 0.1$) cho retention gần $1$. Với $d = 0.4$, retention $\approx 0.91$ theo thực nghiệm trên XAU/USD hourly \cite{de_prado_2018_afml}.

## Công thức

Công thức khai triển đầy đủ cho $d = 0.4$:

$$
(1 - B)^{0.4} X_t \;=\; X_t - 0.4 X_{t-1} - 0.12 X_{t-2} - 0.064 X_{t-3} - 0.0416 X_{t-4} - \dots
$$

Công thức đệ quy tính hệ số:

$$
\omega_0 = 1, \qquad \omega_k = -\omega_{k-1} \cdot \frac{d - k + 1}{k}, \; k \geq 1.
$$

## Tham số quan trọng

| Tham số | Giá trị | Lý do |
|---|---|---|
| $d$ (bậc vi phân) | $0.4$ | ADF $p$-value $< 0.05$, variance retention $\approx 0.91$ |
| $\epsilon$ (ngưỡng cắt) | $10^{-4}$ | Số hệ số $\sim 30$, đủ memory mà không overfit lag xa |
| Kiểm định dừng | ADF | Tiêu chuẩn cho tài chính \cite{hamilton_1994_time_series} |

Lý do chọn $d = 0.4$ thay vì $d = 0.0$ (giá gốc) hay $d = 1.0$ (return): thử nghiệm trên XAU/USD 1h cho thấy $d < 0.35$ không qua ADF, $d > 0.5$ giảm variance retention dưới $0.85$ mất tín hiệu dự báo. $d = 0.4$ là điểm Sweet spot.

## Kết quả thực nghiệm

Kết quả ADF test trên chuỗi `close_fracdiff` cho XAU/USD hourly 5 năm (đọc từ `reports/run_*/run_data.json`):

- $d = 0.0$: ADF statistic $-1.83$, $p = 0.37$ — không dừng.
- $d = 0.4$: ADF statistic $-5.21$, $p = 8 \cdot 10^{-6}$ — dừng ở mức $1\%$.
- $d = 1.0$: ADF statistic $-12.4$, $p < 10^{-20}$ — dừng nhưng variance retention $\approx 0.03$.

## Tham khảo

- `\cite{de_prado_2018_afml}` — Marcos Lopez de Prado, *Advances in Financial Machine Learning*, Wiley 2018, Ch. 4.
- `\cite{hamilton_1994_time_series}` — James D. Hamilton, *Time Series Analysis*, Princeton 1994, Ch. 4.
