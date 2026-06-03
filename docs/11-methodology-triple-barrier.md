---
doc: 11-methodology-triple-barrier
stage: methodology
thesis_chapter: 2
status: draft
last_updated: 2026-06-02
code_ref: null
---

# Methodology — Triple Barrier

> Phương pháp gán nhãn chuỗi thời gian tài chính qua ba thanh chắn động — take-profit, stop-loss và vertical (thời gian) — giúp khai thác đặc tính path-dependent mà fixed-horizon return labeling bỏ qua.

## Tóm tắt

Triple-barrier labeling, do López de Prado đề xuất \cite{de_prado_2018_afml}, là kỹ thuật gán nhãn cho bài toán dự báo tín hiệu giao dịch trên chuỗi thời gian tài chính. Thay vì gán nhãn theo chiều hướng lợi nhuận tại một thời điểm cố định (return $> 0$ thì $+1$, return $< 0$ thì $-1$), phương pháp đặt ba "thanh chắn" (barrier) dưới dạng mức giá trên, mức giá dưới và thời điểm đóng cửa. Nhãn được xác định bởi barrier nào bị chạm đầu tiên — qua đó nắm bắt thông tin path-dependent của quỹ đạo giá trong cửa sổ dự báo.

## Cơ sở lý thuyết

### Bài toán gán nhãn tài chính

Bài toán học có giám sát trên dữ liệu giá yêu cầu một ánh xạ từ mỗi mẫu đầu vào $x_t$ (vector đặc trưng tại thời điểm $t$) sang một nhãn rời rạc $y_t \in \{-1, 0, +1\}$. Cách tiếp cận truyền thống — *fixed-horizon return labeling* — đặt

$$
y_t \;=\; \mathrm{sign}\bigl( P_{t+h} / P_t - 1 \bigr),
$$

với $h$ là horizon cố định (số nến). Cách này có ba nhược điểm quan trọng \cite{de_prado_2018_afml}:

1. **Bỏ qua path**: hai kịch bản có cùng $P_{t+h}$ nhưng đường đi khác nhau (một cái đi thẳng lên, một cái đảo chiều mạnh rồi phục hồi) được gán cùng nhãn dù đặc tính rủi ro khác hẳn.
2. **Không thích ứng volatility**: cùng horizon $h$, biến động hẹp hay rộng đều trả về cùng định dạng nhãn — không phản ánh regime thị trường.
3. **Mất cân bằng nhãn**: trong thị trường có xu hướng dài hạn, return labeling nghiêng về một phía, làm mô hình học lệch.

### Ba thanh chắn

Tại thời điểm mở position $t$ với giá tham chiếu $P_t$, đặt ba mức barrier:

- **Take-profit (TP)** — mức giá trên:

$$
B^{\mathrm{TP}}_t \;=\; P_t\bigl(1 + \tau^{\mathrm{TP}}_t\bigr).
$$

- **Stop-loss (SL)** — mức giá dưới:

$$
B^{\mathrm{SL}}_t \;=\; P_t\bigl(1 - \tau^{\mathrm{SL}}_t\bigr).
$$

- **Vertical (V)** — thời điểm đóng:

$$
B^{\mathrm{V}}_t \;=\; t + h.
$$

Trong đó $\tau^{\mathrm{TP}}_t, \tau^{\mathrm{SL}}_t > 0$ là **độ rộng động** (dynamic width) — thường được đặt tỷ lệ với độ biến động cục bộ, ví dụ thông qua ATR (Average True Range) hoặc độ lệch chuẩn của return dạng exponential moving average. Cụ thể:

$$
\tau^{\mathrm{TP}}_t \;=\; c^{\mathrm{TP}} \cdot \frac{\mathrm{ATR}_t}{P_t}, \qquad \tau^{\mathrm{SL}}_t \;=\; c^{\mathrm{SL}} \cdot \frac{\mathrm{ATR}_t}{P_t},
$$

với $c^{\mathrm{TP}}, c^{\mathrm{SL}}$ là các hằng số kiểm soát risk-reward ratio.

### Quy tắc gán nhãn

Quan sát quỹ đạo giá $\{P_s\}_{s=t+1}^{t+h}$ trong cửa sổ dự báo. Đặt

$$
t^{*} \;=\; \min \Bigl\{ s \in \{t+1, \dots, t+h\} \;:\; P_s \geq B^{\mathrm{TP}}_t \;\text{hoặc}\; P_s \leq B^{\mathrm{SL}}_t \Bigr\},
$$

là thời điểm đầu tiên mà giá chạm TP hoặc SL. Nhãn được gán theo công thức:

$$
y_t \;=\;
\begin{cases}
+1, & \text{nếu } P_{t^{*}} \geq B^{\mathrm{TP}}_t \quad (\text{TP hit}), \\
-1, & \text{nếu } P_{t^{*}} \leq B^{\mathrm{SL}}_t \quad (\text{SL hit}), \\
\;\;0, & \text{nếu không có barrier bị chạm đến } t + h \quad (\text{vertical hit}).
\end{cases}
$$

Tập nhãn $\{-1, 0, +1\}$ tương ứng với ba kịch bản: stop-loss kích hoạt (thua), vertical hết hạn (không xác định), take-profit đạt (thắng).

### So sánh với fixed-horizon return labeling

Fixed-horizon chỉ có 2 nhãn ($\pm 1$), không path-dependent, không adaptive volatility, không phản ánh risk management, và mất cân bằng trong trend market. Triple-barrier khắc phục tất cả: 3 nhãn (hoặc 2 sau binary mapping), path-dependent, adaptive qua ATR, barrier chính là TP/SL thực tế, balance tốt hơn nhờ dynamic width.

### Ưu điểm chính

1. **Path-dependent**: thứ tự chạm barrier mang thông tin về tính chất của quỹ đạo — quan trọng cho chiến lược giao dịch thực tế vốn bị ràng buộc bởi stop-loss/take-profit.
2. **Adaptive volatility**: barrier mở rộng khi thị trường biến động mạnh, thu hẹp khi thị trường yên tĩnh — giảm nhiễu nhãn trong regime cao volatility.
3. **Tích hợp risk management**: barrier chính là mức TP/SL thực tế mà trader đặt — nhãn trở thành proxy trực tiếp cho kết quả giao dịch, không còn là proxy cho chiều hướng giá.
4. **Linh hoạt qua vertical**: nhãn 0 cho phép loại bỏ các mẫu không quyết định được, hoặc ánh xạ vào một lớp "kịch báo thấp" — giảm false positive cho mô hình phân loại.

### Hạn chế và biến thể

Trong luận văn này, nhãn vertical ($0$) được ánh xạ vào $-1$ (xem `03-labeling-triple-barrier.md`), chuyển bài toán về binary $\{-1, +1\}$ với giả định bảo thủ: cửa sổ dự báo không quyết đoán được coi là tín hiệu thất bại. Nhạy với tham số width — chọn $\tau$ quá nhỏ làm tăng nhiễu, quá lớn mất tín hiệu — cần auto-tune. Tính toán phức tạp hơn return labeling: cần quét mỗi mẫu trong horizon — chi phí $O(n \cdot h)$, song có thể vector hóa với Numba.

## Công thức

Công thức tổng quát cho dynamic width dựa trên ATR:

$$
\boxed{\;
B^{\mathrm{TP}}_t = P_t \cdot \bigl(1 + c^{\mathrm{TP}} \cdot \mathrm{ATR}_t / P_t\bigr), \qquad
B^{\mathrm{SL}}_t = P_t \cdot \bigl(1 - c^{\mathrm{SL}} \cdot \mathrm{ATR}_t / P_t\bigr)
\;}
$$

Thời điểm chạm barrier đầu tiên:

$$
t^{*}(t) \;=\; \inf \Bigl\{ s > t \;:\; \bigl(P_s \geq B^{\mathrm{TP}}_t\bigr) \lor \bigl(P_s \leq B^{\mathrm{SL}}_t\bigr) \Bigr\} \land (t + h).
$$

## Tham số quan trọng

| Tham số | Ký hiệu | Vai trò |
|---|---|---|
| Horizon | $h$ | Số nến cửa sổ dự báo, kiểm soát thời gian chờ đợi |
| TP width | $c^{\mathrm{TP}}$ | Hệ số nhân ATR cho take-profit |
| SL width | $c^{\mathrm{SL}}$ | Hệ số nhân ATR cho stop-loss |
| Volatility estimator | $\mathrm{ATR}_t$ | Đo lường biến động cục bộ, có thể thay bằng EMA của $|r_t|$ |
| Width mode | swing / ATR / auto-tune | Lựa chọn nguồn dynamic width (xem `03-labeling-triple-barrier.md`) |

Lý do chọn ATR làm estimator: ATR đã được chuẩn hóa theo giá, phổ biến trong tài chính \cite{hamilton_1994_time_series}, phản hồi nhanh thay đổi regime thông qua rolling window ngắn (thường 14 nến).

## Kết quả thực nghiệm

Trên dataset XAU/USD hourly 5 năm $\approx 44{,}000$ nến, với $h = 24$, $c^{\mathrm{TP}} \in [0.5, 4.0]$ (auto-tune, fallback 2.0), $c^{\mathrm{SL}} = 1.5$, và dynamic width từ swing H/L (xem `03-labeling-triple-barrier.md`):

- Tỉ lệ nhãn $+1$ (TP hit): $\approx 38\%$.
- Tỉ lệ nhãn $-1$ (SL + vertical hit): $\approx 62\%$.
- Balance ratio (min/max): $\approx 0.61$ — tốt hơn return labeling ($\approx 0.43$) nhờ vào dynamic width.

Chi tiết phân bố nhãn và auto-tune kết quả được trình bày ở `03-labeling-triple-barrier.md`.

## Tham khảo

- `\cite{de_prado_2018_afml}` — Marcos Lopez de Prado, *Advances in Financial Machine Learning*, Wiley 2018, Ch. 3: Meta-labeling and triple-barrier method.
- `\cite{hamilton_1994_time_series}` — ATR, rolling volatility estimators.
- `docs/03-labeling-triple-barrier.md` — chi tiết cài đặt.
- `docs/12-methodology-meta-labeling.md` — meta-labeling bổ sung cho triple-barrier output.
