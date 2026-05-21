# Tổng Quan Toàn Diện và Định Hướng Thực Tiễn: Ứng Dụng Mô Hình Hybrid Stacking Dự Báo Tín Hiệu Giao Dịch CFD Vàng

Sự phát triển của thị trường tài chính định lượng trong những năm gần đây đã chứng kiến một bước chuyển dịch lớn từ các công cụ dự báo đơn lẻ sang các kiến trúc học máy tích hợp đa tầng. Đối với một loại tài sản có tính phi tuyến mạnh, độ biến động cao và chịu ảnh hưởng phức tạp từ nhiều yếu tố vĩ mô như CFD Vàng (XAU/USD), các phương pháp tiếp cận truyền thống thường gặp giới hạn trong việc bao quát toàn bộ các đặc trưng của chuỗi thời gian.

Trong bối cảnh đó, phương pháp xếp chồng hỗn hợp (Hybrid Stacking) nổi lên như một giải pháp đột phá, cho phép kết hợp các thế mạnh bổ trợ của nhiều trường phái thuật toán khác nhau nhằm tối ưu hóa độ chính xác và tính ổn định của tín hiệu giao dịch.

## Bối Cảnh Nghiên Cứu và Tiến Trình Phát Triển của Mô Hình Hybrid Stacking

Lịch sử phát triển của các mô hình dự báo chuỗi thời gian tài chính ghi nhận một xu hướng chuyển dịch rõ rệt từ các thuật toán thống kê cổ điển sang các hệ thống lai (hybrid) và tích hợp (ensemble). Giai đoạn quanh năm 2020 đánh dấu sự trỗi dậy mạnh mẽ của việc kết hợp học máy truyền thống với các kiến trúc học sâu nhằm khai thác tốt hơn các phụ thuộc thời gian và log hoạt động của thị trường.

Đến năm 2022, xu hướng này tiến hóa sâu hơn vào các cấu trúc xếp chồng hai lớp (two-layer) và đa giai đoạn (multi-stage), giúp đa dạng hóa mô hình và cải thiện khả năng chống quá khớp (overfitting) trên các tập dữ liệu có độ nhiễu cao.

Mặc dù có sự tiến bộ vượt bậc, các mô hình dự báo tài chính vẫn phải đối mặt với ba thách thức cốt lõi chưa được giải quyết triệt để trong các nghiên cứu đơn lẻ:

1. Sự thiếu hụt trong việc tích hợp đồng thời các thuộc tính tĩnh (static attributes) và các mô hình hành vi động (dynamic behavioral patterns).
2. Sự hạn chế trong việc nghiên cứu sâu về các cơ chế tương tác lai giữa thuật toán học máy truyền thống (ML) và học sâu (DL).
3. Tính thiếu minh bạch và khả năng giải thích (interpretability) kém của các mạng nơ-ron sâu trong môi trường ra quyết định tài chính.

Để giải quyết các vấn đề này, kiến trúc Hybrid Stacking được phát triển nhằm thiết lập một khung làm việc thống nhất, không rò rỉ dữ liệu (leakage-proof), kết hợp hài hòa giữa hiệu năng dự báo và ý nghĩa thực tiễn trong quản trị rủi ro.

## Cơ Sở Toán Học và Kiến Trúc Phân Tầng của Hệ Thống Hybrid Stacking

Phương pháp xếp chồng (Stacking Generalization) hoạt động dựa trên nguyên lý huấn luyện một tập hợp các bộ học cơ sở (Level 0 Base Learners) độc lập, sau đó sử dụng các kết quả dự báo ngoài phân đoạn (out-of-fold predictions) của chúng làm dữ liệu đầu vào để huấn luyện một bộ siêu học (Level 1 Meta-learner). Cơ chế phân tầng này giúp hệ thống tận dụng tối đa thế mạnh của từng thuật toán thành phần, đồng thời giảm thiểu sai số chệch (bias) và phương sai (variance) của toàn bộ hệ thống.

Ở phân tầng Level 0, các mạng nơ-ron hồi quy như LSTM (Long Short-Term Memory) và GRU (Gated Recurrent Unit) thường được sử dụng nhờ khả năng xử lý chuỗi dữ liệu tuần tự và khắc phục hiện tượng tiêu biến gradient. Kiến trúc của mạng LSTM được kiểm soát bởi các hàm cổng đặc trưng giúp lưu trữ thông tin dài hạn:

$$
f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)
$$

$$
i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)
$$

$$
\tilde{C}_t = \tanh(W_C[h_{t-1}, x_t] + b_C)
$$

$$
o_t = \sigma(W_o[h_{t-1}, x_t] + b_o)
$$

$$
h_t = o_t \odot \tanh(C_t)
$$

Trong khi đó, mạng GRU tối ưu hóa hiệu năng tính toán thông qua cấu trúc cổng đơn giản hơn bao gồm cổng đặt lại ($r_t$) và cổng cập nhật ($z_t$):

$$
r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)
$$

$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)
$$

$$
\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])
$$

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

Để hỗ trợ các bộ học sâu trích xuất tín hiệu hiệu quả từ biểu đồ nến XAU/USD, các chỉ báo kỹ thuật toán học được tính toán và đưa vào không gian đặc trưng. Bảng dưới đây tổng hợp các công thức toán học của các chỉ báo kỹ thuật phổ biến được sử dụng trong các hệ thống dự báo tài chính:

| Tên Chỉ báo Kỹ thuật | Ký hiệu | Công thức Toán học | Ý nghĩa trong Dự báo Tín hiệu |
| --- | --- | --- | --- |
| Exponential Moving Average | $EMA_t$ | $\frac{2}{N+1} \times Close_t + \left(1 - \frac{2}{N+1}\right) \times EMA_{t-1}$ | Xác định xu hướng giá mượt mà, giảm thiểu độ trễ so với SMA. |
| Moving Average Convergence Divergence | $MACD$ | $EMA_{12} - EMA_{26}$ | Đo lường động lượng và xác định các điểm đảo chiều xu hướng. |
| Relative Strength Index | $RSI$ | $100 - \frac{100}{1 + \frac{\text{Avg Gain}}{\text{Avg Loss}}}$ | Xác định trạng thái quá mua hoặc quá bán của thị trường. |
| Stochastic Oscillator | $SO$ | $\frac{Close_t - Low_{\text{min}}}{High_{\text{max}} - Low_{\text{min}}} \times 100$ | So sánh giá đóng cửa với phạm vi giá trong một khoảng thời gian nhất định. |

## Đánh Giá các Nghiên Cứu Học Thuật về Mô Hình Stacking trong Tài Chính và Dự Báo Vàng

Các công trình nghiên cứu học thuật quốc tế đã chứng minh tính hiệu quả của mô hình Stacking thông qua nhiều cấu trúc tích hợp khác nhau, từ các hệ thống học sâu thuần túy đến các mô hình lai đa phương thức.

### Hệ Thống Stacking Học Sâu Hai Chiều: BiLSTM-BiGRU-RF

Một nghiên cứu điển hình đã đề xuất hệ thống Stacking sử dụng hai mạng nơ-ron hồi quy hai chiều làm bộ học cơ sở: BiLSTM và BiGRU, kết hợp với bộ siêu học là Rừng ngẫu nhiên (Random Forest). Việc xử lý dữ liệu theo cả hai hướng tiến và lùi giúp mô hình khai thác trọn vẹn ngữ cảnh lịch sử của chuỗi giá vàng.

Được thử nghiệm trên tập dữ liệu vàng từ ngày 1 tháng 1 năm 2020 đến ngày 31 tháng 5 năm 2024 với các khoảng lookback (7, 15, và 30 ngày), mô hình này đã đạt được hiệu năng ấn tượng với sai số cực thấp. Tại khoảng lookback 30 ngày, hệ thống ghi nhận các chỉ số tối ưu bao gồm sai số bình phương trung bình cực tiểu ($MSE = 0.000$), sai số tuyệt đối trung bình ($MAE = 0.0050$), và hệ số xác định ($R^2 = 0.9984$), khẳng định ưu thế tuyệt đối của việc xếp chồng hai chiều so với các mạng nơ-ron đơn lẻ.

### Mô Hình Stacking Hỗn Hợp Đa Dạng: ARIMA-ML-DL-XGBoost

Nhằm giải quyết tính phi tĩnh của chuỗi thời gian tài chính, một hướng nghiên cứu khác đã tích hợp thành công mô hình thống kê ARIMA cùng các bộ học học máy và học sâu (Random Forest, GRU, LSTM, Transformer) ở phân tầng Level 0, sử dụng thuật toán tăng cường độ dốc cực đại XGBoost làm bộ siêu học ở Level 1. XGBoost xây dựng các cây quyết định một cách tuần tự để sửa chữa sai số của các cây trước đó, tối ưu hóa hàm mục tiêu có chứa thành phần chuẩn hóa để kiểm soát hiện tượng quá khớp.

Khi thực nghiệm trên các chỉ số tài chính và giá cổ phiếu phi tĩnh, hệ thống này đạt độ chính xác định hướng xu hướng (Directional Accuracy) vượt trội từ $0.84$ đến $0.92$, cùng chỉ số nhận diện đảo chiều xu hướng (Trend Reversal Identification) đạt $0.68$ đến $0.87$. Kết quả phân tích phân rã (ablation studies) cho thấy việc loại bỏ mô hình thống kê ARIMA hoặc mô hình Transformer khỏi phân tầng Level 0 đều làm suy giảm đáng kể độ chính xác và khả năng tổng quát hóa của hệ thống.

### Khung Phân Tầng Quản Trị Rủi Ro Đa Lớp: TL-StackLR

Một cách tiếp cận có giá trị tham chiếu cao cho việc dự báo tín hiệu giao dịch là mô hình TL-StackLR (Transformer-LightGBM-Stacked Logistic Regression). Ban đầu được thiết kế cho việc đánh giá rủi ro tín dụng đa lớp, mô hình này sử dụng cấu trúc Feature Tokenizer Transformer (FT-Transformer) để nắm bắt các tương tác đặc trưng dạng bảng, kết hợp với LightGBM cho các mẫu phi tuyến tính, và đưa qua bộ siêu học hồi quy Logistic để hiệu chuẩn xác suất đầu ra.

Điểm đặc biệt của mô hình này là việc chuyển đổi bài toán nhị phân đơn giản thành phân loại ba trạng thái (Low, Medium, High Risk) dựa trên phân tầng lượng phân vị. Khi áp dụng vào dự báo CFD vàng, cơ chế này có thể chuyển đổi trực tiếp thành các tín hiệu giao dịch thực tế tương ứng: Bán (Short), Đứng ngoài (Hold), và Mua (Long), đồng thời cung cấp khả năng giải thích trực quan thông qua công cụ SHAP (SHapley Additive exPlanations).

### Hiện Tượng Quá Khớp của Học Sâu trên Dữ Liệu Mẫu Nhỏ

Một phát hiện quan trọng trong nghiên cứu thực nghiệm đối sánh hệ thống Stacking (RF+XGB+SVR) và mô hình Hybrid sửa lỗi residual (LSTM+SVRres) trên các dữ liệu hàng hóa là xu hướng quá khớp của các kiến trúc học sâu. Trong bài toán dự báo Titanium PPI vĩ mô hàng tháng (giai đoạn 2018-2025), thuật toán Rừng ngẫu nhiên (Random Forest) đơn giản lại là mô hình hoạt động tốt nhất.

Lý do là vì các mô hình học sâu tuần tự phức tạp như LSTM đòi hỏi lượng dữ liệu cực lớn để hội tụ; khi huấn luyện trên tập dữ liệu mẫu nhỏ hoặc có độ nhiễu regime cao, chúng dễ bị phạt bởi hiện tượng quá khớp và có xu hướng bỏ qua các sự kiện khủng hoảng hiếm gặp. Bảng dưới đây thể hiện chi tiết kết quả đối sánh hiệu năng dự báo ngoài mẫu giữa các thuật toán trên chuỗi chỉ số Titanium PPI:

| Thuật toán / Mô hình | Sai số Tuyệt đối Trung bình (MAE) | Sai số Bình phương Trung bình (RMSE) | Đánh giá Khả năng Ứng dụng Thực tế |
| --- | ---: | ---: | --- |
| Random Forest (RF) | 0.035975 | 0.045313 | Tối ưu nhất trên dữ liệu mẫu nhỏ và phi tĩnh. |
| XGBoost | 0.070187 | 0.088476 | Hiệu năng trung bình, dễ bị ảnh hưởng bởi tham số. |
| Support Vector Regression (SVR) | 0.112631 | 0.150640 | Độ lệch lớn khi xu hướng thị trường thay đổi nhanh. |
| LSTM | 0.100452 | 0.120770 | Quá khớp mạnh do thiếu hụt kích thước mẫu huấn luyện. |
| Hybrid (LSTM + SVRres) | 0.266693 | 0.324666 | Sai số tích lũy từ hai giai đoạn huấn luyện độc lập. |

## Hiện Trạng Khai Thác Mã Nguồn Mở và Các Dự Án Thực Tế trên GitHub

Để tìm kiếm các công cụ hiện thực hóa mô hình Hybrid Stacking cho CFD Vàng, nền tảng GitHub cung cấp một số kho lưu trữ mã nguồn mở tiêu biểu với các hướng tiếp cận thực tiễn khác nhau.

### Dự Án ayoub-mg / Gold-Price-Forecasting

Dự án này tập trung hoàn toàn vào việc thiết lập các mô hình dự báo giá vàng và đánh giá hiệu quả chiến lược giao dịch đi kèm.

- Cơ chế Stacking: Tác giả kết hợp các dự báo từ bốn thuật toán học máy ở Level 0 bao gồm AdaBoost, Random Forest, Gradient Boosting, và Extra Trees. Dự báo của các bộ học này sau đó làm đầu vào cho mô hình XGBoost ở phân tầng siêu học Level 1.
- Kết quả thực nghiệm: Đáng chú ý, mô hình Stacking này chỉ đạt hệ số xác định $R^2 = 0.445287$ và Sharpe Ratio là $0.6$. Trong khi đó, mô hình LSTM độc lập trong cùng dự án đạt $R^2 = 0.978041$ và Sharpe Ratio là $0.81$.

Kết quả này chỉ ra một bài học thực tế quan trọng: việc xếp chồng các mô hình có cấu trúc quá tương đồng (đều là các thuật toán dựa trên cây quyết định ở Level 0) không tạo ra đủ tính đa dạng đặc trưng để bộ siêu học XGBoost tối ưu hóa, dẫn đến hiệu năng kém hơn cả một mô hình học sâu đơn lẻ.

### Hệ Thống Giao Dịch Tự Động DART (Deep Adaptive Reinforcement Trader)

DART đại diện cho một hệ thống giao dịch thuật toán chuyên nghiệp được thiết kế dưới dạng một nền tảng lai ghép đa phương thức.

- Kiến trúc Lai (Hybrid Architecture): Hệ thống tích hợp một tác nhân học sâu củng cố Soft Actor-Critic (SAC) với mô hình xếp chồng học máy truyền thống.
- Triển khai Stacking: Module `ml/trading_ai.py` thực hiện xếp chồng bộ học cơ sở Gradient Boosting và Rừng ngẫu nhiên để dự đoán xu hướng, sau đó sử dụng thuật toán hồi quy Logistic (Logistic Regression) làm bộ siêu học để đưa ra quyết định giao dịch cuối cùng.
- Đặc trưng nâng cao: Khung làm việc này sử dụng mô hình Mạng tự mã hóa biến phân (VAE) để nhận diện 7 trạng thái thị trường (market regimes) khác nhau, đồng thời định lượng độ bất định thông qua sự bất đồng thuận giữa các mô hình trong ensemble để kiểm soát tỷ lệ phân bổ vốn Kelly và rủi ro VaR.

### Khung Tích Hợp Đa Nguồn Amizaa / gold-price-prediction

Kho lưu trữ này cung cấp giải pháp dự báo giá vàng bằng cách kết hợp phân tích định lượng chuỗi thời gian và phân tích định tính ngữ nghĩa tin tức.

- Kiến trúc hệ thống: Dự án kết hợp mô hình ARIMA cổ điển, mạng học sâu LSTM (đầu vào là chuỗi giá OHLC 60 ngày, đạt $R^2 = 0.978$ trên tập thử nghiệm), và mô hình khuếch tán (Diffusion Models) để mô phỏng chuỗi giá.
- Phân tích ngữ nghĩa tin tức: Hệ thống tự động thu thập tin tức từ các trang BullionVault, Kitco, FXStreet, tiến hành phân cụm từ khóa bằng K-means và sử dụng các mô hình ngôn ngữ lớn (LLM) như Seed-OSS-36B và Qwen3-32B để dự báo tác động của tin tức đối với giá vàng.

Kết quả thử nghiệm cho thấy việc tích hợp phân tích ngữ nghĩa qua LLM giúp đạt độ chính xác $62.2\%$ trong việc dự báo hướng đi của giá vàng.

### Khung Kiểm Định livealgos và Định Giá Walk-Forward

Dự án livealgos cung cấp một mẫu thiết kế Stacking hỗn hợp hoàn chỉnh cho giao dịch trực tuyến: MLP, VAE, LSTM, XGBoost, LGBM, CatBoost, RF $\rightarrow$ Logistic Regression.

Dự án giải quyết triệt để vấn đề rò rỉ dữ liệu thông qua cơ chế kiểm định cuộn bước tiếp (Walk-Forward Validation). Đây là phương pháp kiểm định bắt buộc đối với dữ liệu tài chính nhằm duy trì tính tuần tự thời gian, tránh việc mô hình sử dụng dữ liệu tương lai để dự báo quá khứ. Dự án cũng cung cấp hơn 20 thuật toán giảm chiều dữ liệu (như PCA) để lọc ra 100 đặc trưng tối ưu từ hơn 11,000 biến đầu vào ban đầu.

### So Sánh các Đặc Điểm Kỹ Thuật của các Dự Án Thực Tế trên GitHub

Bảng dưới đây hệ thống hóa các đặc điểm kỹ thuật của các dự án mã nguồn mở tiêu biểu trên GitHub liên quan trực tiếp đến cấu trúc lai và xếp chồng trong giao dịch tài chính:

| Tên Dự án (GitHub) | Ngôn ngữ / Công cụ | Mô hình Cấu trúc (Level 0 → Level 1) | Nguồn Dữ liệu & Đặc trưng Đầu vào | Điểm nổi bật & Tính năng Đặc thù |
| --- | --- | --- | --- | --- |
| ayoub-mg / Gold-Price-Forecasting | Python / Jupyter Notebook | AdaBoost, RF, ExtraTrees, GBDT $\rightarrow$ XGBoost | Giá vàng lịch sử, các chỉ báo kỹ thuật (SMA, EMA, ATR, RSI) | Triển khai chiến lược giao dịch dựa trên tín hiệu dự báo. |
| ItzSwapnil / DART | Python / PyTorch / FastAPI | Gradient Boosting, Random Forest $\rightarrow$ Hồi quy Logistic | 50+ Chỉ báo kỹ thuật, cấu trúc thị trường, Sentiment | Tích hợp Deep RL (SAC), Nhận diện trạng thái thị trường bằng VAE. |
| Amizaa / gold-price-prediction | Jupyter Notebook / Python | ARIMA, LSTM, Diffusion Models, LLM-based Sentiment | Giá lịch sử MetaTrader (2004-2025), Tin tức BullionVault, Kitco, FXStreet | Phân tích tâm lý tin tức bằng LLM, phân cụm từ khóa K-means. |
| impulsecorp / livealgos | Python / Ubuntu | MLP, VAE, LSTM, XGBoost, LGBM, CatBoost, RF $\rightarrow$ Hồi quy Logistic | Dữ liệu nến giao dịch trực tuyến | Walk-Forward Validation, tự động lọc đặc trưng từ 11,000+ xuống 100 biến. |

## Ý Nghĩa Thực Tiễn, Tích Hợp Ý Kiến Thị Trường và Thách Thức trong Giao Dịch CFD Vàng

Việc áp dụng các mô hình Hybrid Stacking vào giao dịch thực tế trên sàn giao dịch CFD (như MT5) đòi hỏi sự hiểu biết sâu sắc về các yếu tố vận hành thực tế của thị trường, vượt ra ngoài các chỉ số sai số toán học thuần túy.

### Sự Khác Biệt giữa Thử Nghiệm và Giao Dịch Thực Tế

Trong môi trường backtest, các mô hình học sâu thường báo cáo độ chính xác rất cao. Tuy nhiên, khi đưa vào giao dịch trực tuyến (live trading), hiệu suất của hệ thống thường bị suy giảm nghiêm trọng do các yếu tố kỹ thuật:

- Độ lệch giá (Slippage) và Chi phí Giao dịch: CFD vàng là một trong những tài sản có tính thanh khoản cao nhưng khoảng giãn spread và phí qua đêm (swap) có thể tăng đột biến trong các kỳ công bố tin tức vĩ mô. Các mô hình Stacking tạo tín hiệu tần suất cao nếu không được tối ưu hóa chi phí giao dịch sẽ dễ dàng làm xói mòn toàn bộ lợi nhuận lý thuyết.
- Độ trễ xử lý (Execution Latency): Việc chạy một hệ thống Hybrid Stacking bao gồm cả mạng LSTM, GRU và mô hình cây quyết định yêu cầu tài nguyên tính toán lớn. Nếu thời gian suy luận (inference time) vượt quá vài giây, mức giá khớp lệnh thực tế trên nền tảng MT5 sẽ bị sai lệch so với mức giá dự báo của mô hình, làm vô hiệu hóa điểm vào lệnh (entry point) tối ưu.

### Tầm Quan Trọng của Việc Tích Hợp Ý Kiến Thị Trường và Dữ Liệu Đa Phương Thức

Thị trường vàng chịu sự dẫn dắt mạnh mẽ bởi tâm lý nhà đầu tư và các kỳ vọng lạm phát. Các nghiên cứu chỉ ra rằng các mô hình chỉ sử dụng dữ liệu giá lịch sử thuần túy thường có độ chính xác giới hạn quanh mức $50\%$. Tuy nhiên, khi tích hợp thêm các dữ liệu về độ biến động ngầm ẩn (implied volatility), cường độ giao dịch (trading intensity), và điểm số tâm lý tin tức (FinBERT sentiment score), độ chính xác của mô hình phân loại xu hướng được nâng lên đáng kể, đạt khoảng $70\% - 72\%$.

Việc bổ sung các chỉ báo tâm lý này giúp hệ thống Stacking nhận diện sớm các giai đoạn hưng phấn quá đà hoặc hoảng loạn của thị trường, từ đó hiệu chuẩn lại các tín hiệu giao dịch được tạo ra bởi các bộ học kỹ thuật.

## Định Hướng Chiến Lược và Khung Thực Thi cho Đồ Án Tốt Nghiệp

Để xây dựng một đồ án tốt nghiệp xuất sắc và có tính thực tiễn cao về chủ đề "Ứng dụng mô hình Hybrid Stacking dự báo tín hiệu giao dịch CFD vàng", cấu trúc hệ thống nên được thiết kế bài bản theo các giai đoạn tích hợp sau.

### Thiết Kế Cấu Trúc Phân Tầng Tối Ưu

Khuyến nghị thiết kế hệ thống hai tầng (Level 0 và Level 1) kết hợp giữa các đặc trưng đa dạng:

**Level 0 (Base Learners):**

- Mô hình thống kê: ARIMA (để bắt giữ các quy luật tuyến tính và xu hướng cơ bản).
- Mô hình học máy: LightGBM và Random Forest (để xử lý nhanh các đặc trưng phi tuyến dạng bảng, ổn định hóa phương sai và khai thác các chỉ báo kỹ thuật mà không cần tài nguyên tính toán quá lớn).
- Mô hình học sâu: LSTM (để trích xuất các phụ thuộc chuỗi thời gian dài hạn từ lịch sử giá).
- Mô hình siêu học: Logistic Regression (để hiệu chuẩn xác suất đầu ra của LSTM, LightGBM và Random Forest trong tầng stacking cuối).

**Level 1 (Meta-learner):**

- Sử dụng một mô hình có cấu trúc đơn giản, có tính đơn điệu hoặc được ràng buộc chặt chẽ để chống quá khớp, chẳng hạn như Hồi quy Logistic (Logistic Regression) cho bài toán phân loại tín hiệu (Mua/Bán/Đứng ngoài), hoặc Hồi quy Lasso/Ridge cho bài toán hồi quy giá.

### Xây Dựng Không Gian Đặc Trưng (Feature Space)

Dữ liệu đầu vào cần được mở rộng vượt ra ngoài giá đóng cửa đơn thuần:

- Dữ liệu OHLCV cơ bản của khung thời gian giao dịch chính (ví dụ: H1 hoặc H4) và các khung thời gian lớn hơn (D1) để định hình xu hướng chủ đạo.
- Nhóm chỉ báo kỹ thuật đa dạng: Chỉ báo xu hướng (EMA, MACD), chỉ báo dao động (RSI, Stochastic Oscillator), và chỉ báo biến động (ATR, Bollinger Bands).
- Biến vĩ mô liên kết: Tích hợp chỉ số Dollar Mỹ (DXY), Lợi suất trái phiếu chính phủ Mỹ, và điểm số tâm lý tin tức tài chính được trích xuất từ các mô hình ngôn ngữ lớn chuyên biệt như FinBERT.

### Quy Trình Đánh Giá Mô Hình Chặt Chẽ

Áp dụng nghiêm ngặt phương pháp kiểm định Walk-Forward Validation để mô phỏng chính xác điều kiện giao dịch thực tế.

Thay vì chỉ báo cáo các sai số toán học thuần túy (MSE, RMSE, MAPE), đồ án cần xây dựng một trình mô phỏng giao dịch (Trading Simulator) để tính toán các chỉ số hiệu năng tài chính cốt lõi bao gồm: Tỷ suất lợi nhuận thu được, Sharpe Ratio, Sortino Ratio, và Mức sụt giảm tài sản lớn nhất (Maximum Drawdown). Đây chính là yếu tố quyết định tính thực tiễn và nâng cao giá trị khoa học của đồ án tốt nghiệp.
