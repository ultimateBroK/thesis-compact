---
doc: 90-glossary
stage: glossary
thesis_chapter: B
status: draft
last_updated: 2026-06-02
code_ref: null
---

# Glossary — Phụ lục B

> Danh mục thuật ngữ VI/EN dùng trong luận văn, kèm định nghĩa ngắn và tham chiếu doc liên quan.

## Cơ sở lý thuyết

Luận văn sử dụng nhiều thuật ngữ chuyên ngành từ ba lĩnh vực: (i) tài chính lượng tử (triple-barrier, embargo, PnL), (ii) machine learning (stacking, F1, OOF), (iii) trading kỹ thuật (ATR, RSI, MACD). Bảng dưới liệt kê alphabetically, định nghĩa bằng tiếng Việt, kèm thuật ngữ tiếng Anh gốc và reference doc nội bộ.

## Thuật ngữ

| Thuật ngữ | Viết tắt | Định nghĩa VN | EN term | Ref doc |
|---|---|---|---|---|
| **ADF** | ADF | Test xác nhận tính stationary của time series — kiểm tra unit root. | Augmented Dickey-Fuller test | `10-methodology-fracdiff.md` |
| **ATR** | ATR | Biên độ dao động trung bình trong 14 bar — dùng scale TP/SL barrier. True Range = $\max(H-L, |H-C_{\text{prev}}|, |L-C_{\text{prev}}|)$. | Average True Range | `03-labeling-triple-barrier.md` |
| **Backtest** | — | Mô phỏng chiến lược giao dịch trên dữ liệu quá khứ, đo lường PnL và risk metric. | Backtest | `06-backtest.md` |
| **Calmar ratio** | — | Tỷ số $CAGR / \|MDD\|$ — đo return hàng năm trên worst drawdown. Cao hơn = tốt hơn. | Calmar ratio | `22-evaluation-metrics.md` |
| **Drawdown** | — | Khoảng giảm từ peak cao nhất của equity đến giá trị hiện tại. | Drawdown | `22-evaluation-metrics.md` |
| **Embargo** | — | Buffer khoảng thời gian sau fold validation, loại bỏ khỏi fold train tiếp theo để tránh leakage. | Embargo | `13-methodology-purged-cv.md` |
| **F1 score** | F1 | Trung bình harmonique của precision và recall: $F_1 = 2PR/(P+R)$. Macro = trung bình trên các class. | F1 score | `22-evaluation-metrics.md` |
| **Fractional differencing** | — | Phép sai phân với $d \in (0, 1)$ — loại bỏ trend nhưng giữ memory nhiều hơn so với differencing nguyên. | Fractional differencing | `10-methodology-fracdiff.md` |
| **GRU** | GRU | Variant RNN có gated mechanism, nhẹ hơn LSTM, dùng cho sequence learning. | Gated Recurrent Unit \cite{cho_2014_gru} | `05-models-stacking.md` |
| **LightGBM** | — | Gradient boosting decision tree hiệu năng cao, dùng làm base learner chính. | LightGBM \cite{ke_2017_lightgbm} | `05-models-stacking.md` |
| **Maximum drawdown** | MDD | Drawdown lớn nhất quan sát được trên toàn kỳ backtest: $\min_t (E_t - P_t)/P_t$ với $P_t = \max_{s \le t} E_s$ (running max). | Maximum drawdown | `22-evaluation-metrics.md` |
| **Meta-labeling** | — | Technique học mô hình phụ dự đoán "primary model đúng không", dùng position sizing. | Meta-labeling \cite{kearns_2019_meta} | `12-methodology-meta-labeling.md` |
| **OOF** | OOF | Dự báo của base learner trên fold validation trong cross-validation — dùng train meta-learner và đánh giá chất lượng base. | Out-of-Fold prediction | `14-methodology-stacking.md` |
| **PnL** | PnL | Tổng lợi nhuận/thua lỗ (realized + unrealized) — Profit and Loss. | Profit and Loss | `22-evaluation-metrics.md` |
| **Position sizing** | — | Quyết định khối lượng (lots) vào lệnh dựa trên risk per trade và stop distance. | Position sizing | `06-backtest.md` |
| **Profit factor** | PF | Tỷ số tổng profit / |tổng loss|; $>1$ ⇔ lãi. | Profit factor | `22-evaluation-metrics.md` |
| **Purged k-fold** | — | Cross-validation cho time series: loại bỏ (purge) khoảng overlap label giữa train và val, cộng embargo chống leakage. | Purged k-fold \cite{de_prado_2018_cross_val} | `13-methodology-purged-cv.md` |
| **Sharpe ratio** | — | Tỷ số $\sqrt{N}\mu_R/\sigma_R$ — return annualized trên volatility. Risk-free giả định 0. | Sharpe ratio | `22-evaluation-metrics.md` |
| **Sortino ratio** | — | Variant Sharpe chỉ dùng downside deviation — công bằng hơn với upside vol. | Sortino ratio | `22-evaluation-metrics.md` |
| **DSR** | DSR | Deflated Sharpe Ratio — điều chỉnh Sharpe cho số lần backtest (multiple testing). DSR > 0 ⇔ Sharpe vượt kỳ vọng ngẫu nhiên. | Deflated Sharpe Ratio \cite{de_prado_2018_backtest} | `22-evaluation-metrics.md` |
| **ROC-AUC** | AUC | Area Under ROC Curve — đo phân biệt class; 1.0 = perfect, 0.5 = random. | Receiver Operating Characteristic — AUC | `22-evaluation-metrics.md` |
| **Walk-forward** | — | Kỹ thuật backtest slide window: train trên quá khứ, test trên tương lai, lặp qua thời gian — tránh lookahead bias. | Walk-forward optimization | `06-backtest.md` |
| **Trend filter** | — | Bộ lọc chỉ cho phép trade khi trend rõ (EMA + ADX); tránh sideways regime. | Trend filter | `06-backtest.md` |
| **SHORT_LOT_SCALE** | — | Hệ số scale lot cho SHORT trade (thường < 1) do spread asymmetry; giảm rủi ro directional bias. | Short lot scaling factor | `08-config.md` |
| **Spread** | — | Chênh lệch giá bid/ask tại bar — chi phí giao dịch. | Spread (bid-ask) | `01-data-pipeline.md` |
| **Stacking ensemble** | — | Kết hợp dự báo từ nhiều base learner qua meta-learner — target giảm variance và bias. | Stacking ensemble \cite{wolpert_1992_stacking} | `14-methodology-stacking.md` |
| **SVC** | SVC | Support Vector Machine với kernel RBF — base learner thứ ba. | Support Vector Classifier | `05-models-stacking.md` |
| **Take-profit** | TP | Barrier chốt lời — đóng trade khi giá chạm mức target. | Take-profit | `11-methodology-triple-barrier.md` |
| **Stop-loss** | SL | barrier cắt lỗ — đóng trade khi giá chạm mức stop. | Stop-loss | `11-methodology-triple-barrier.md` |
| **Triple-barrier** | TB | Phương pháp labeling dùng 3 barrier (TP, SL, vertical) — barrier nào chạm trước quyết định label. | Triple-barrier method \cite{de_prado_2018_afml} | `03-labeling-triple-barrier.md`, `11-methodology-triple-barrier.md` |
| **Vertical barrier** | — | Barrier thời gian — đóng trade sau $H$ bar bất kể giá. Label = forward return sign. | Vertical barrier | `11-methodology-triple-barrier.md` |
| **Win rate** | — | Tỷ lệ trade có PnL > 0 trên tổng trade. | Win rate | `22-evaluation-metrics.md` |

## Tham chiếu chéo

Mỗi thuật ngữ link tới doc chi tiết hơn trong cùng `docs/`. Khi gặp thuật ngữ trong text chính, reader có thể tra bảng này rồi follow link tới methodology hoặc module doc.

## Tham khảo

- \cite{de_prado_2018_afml} — López de Prado, methodology tổng hợp.
- \cite{de_prado_2018_cross_val} — purged k-fold + embargo.
- \cite{wolpert_1992_stacking} — stacking generalization.
- \cite{cho_2014_gru} — GRU.
- \cite{ke_2017_lightgbm} — LightGBM.
- \cite{kearns_2019_meta} — meta-labeling for position sizing.
