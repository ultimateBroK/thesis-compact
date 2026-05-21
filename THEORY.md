# Ứng Dụng Mô Hình Hybrid Stacking Dự Báo Tín Hiệu Giao Dịch CFD Vàng: Phân Tích Cấu Trúc, Lý Thuyết Và Quy Trình Triển Khai Thực Nghiệm

## Cấu Trúc Đề Tài Và Bản Chất Thị Trường CFD Vàng

Đề tài "Ứng dụng mô hình Hybrid Stacking dự báo tín hiệu giao dịch CFD vàng" đại diện cho một nghiên cứu mang tính liên ngành sâu sắc, kết hợp giữa toán học tài chính, cấu trúc vi mô thị trường và khoa học dữ liệu hiện đại. Việc phân tích cấu trúc tên đề tài đòi hỏi sự phân rã chi tiết thành ba thành phần cốt lõi: đối tượng tài chính giao dịch, mục tiêu dự báo đầu ra và phương pháp luận học máy xếp chồng hỗn hợp được đề xuất.

Hợp đồng chênh lệch vàng (Gold CFD - ký hiệu phổ biến là XAU/USD) là một trong những sản phẩm tài chính có tính thanh khoản cao nhất toàn cầu, đồng thời cũng sở hữu mức độ biến động vô cùng phức tạp. Vàng không chỉ đóng vai trò là một hàng hóa vật chất mà còn là một tài sản trú ẩn an toàn, một công cụ phòng hộ lạm phát và là hàn thử biểu đo lường sức khỏe của nền kinh tế vĩ mô. Sự biến động của giá vàng chịu ảnh hưởng đồng thời bởi các yếu tố cấu trúc như lãi suất thực tế Mỹ, sức mạnh đồng USD (chỉ số DXY), chính sách tiền tệ của các ngân hàng trung ương, rủi ro địa chính trị toàn cầu và động lực cung cầu thị trường. Sự tương tác của các biến số này tạo ra các đặc tính phi tuyến tính mạnh mẽ, hiện tượng phương sai thay đổi theo thời gian (volatility clustering) và các sự kiện dịch chuyển trạng thái thị trường (regime shifts) đột ngột. Điều này khiến việc áp dụng các mô hình kinh tế lượng tuyến tính truyền thống trở nên kém hiệu quả.

Khác với các nghiên cứu dự báo giá đóng cửa thuần túy thuộc bài toán hồi quy, mục tiêu của đề tài này là dự báo "tín hiệu giao dịch". Sự chuyển dịch từ dự báo giá trị liên tục sang dự báo hành động phân loại (Mua - Buy, Bán - Sell, Đứng ngoài - Neutral) phản ánh tư duy thực tế của một hệ thống giao dịch định lượng (quantitative trading system). Tín hiệu giao dịch không chỉ đơn thuần dựa vào xu hướng giá mà còn phải tích hợp các quy tắc quản trị rủi ro nghiêm ngặt bao gồm điểm vào lệnh (entry), điểm cắt lỗ (stop loss), và điểm chốt lời (take profit) trong một khung thời gian xác định. Do đó, bài toán dự báo ở đây cần được định nghĩa dưới dạng phân loại đa lớp (multi-class classification) hoặc cấu trúc dự báo mục tiêu linh hoạt nhằm nắm bắt các đặc trưng động lượng khác nhau của thị trường.

Mô hình học máy xếp chồng hỗn hợp (Hybrid Stacking) là trung tâm công nghệ của đề tài này. Định hướng "Hybrid" thể hiện việc kết hợp các trường phái học máy khác nhau nhằm tận dụng thế mạnh đặc thù của từng kiến trúc. Các mô hình học sâu chuỗi thời gian như LSTM, GRU hay Transformer có khả năng học các biểu diễn phụ thuộc thời gian dài hạn và các đặc trưng động từ chuỗi giá lịch sử. Ngược lại, các thuật toán dựa trên cây quyết định tăng cường độ dốc (GBDT) như XGBoost, LightGBM và CatBoost lại cực kỳ xuất sắc trong việc khai thác mối quan hệ tương tác phi tuyến tính phức tạp giữa các đặc trưng dạng bảng tĩnh hoặc các chỉ báo kỹ thuật được tính toán thủ công. Phương pháp Stacking cho phép xây dựng một kiến trúc hai tầng. Tầng cơ sở (Level 0) chứa các mô hình đa dạng (heterogeneous models) hoạt động độc lập. Dự báo đầu ra của tầng này trở thành thuộc tính đầu vào để huấn luyện một mô hình siêu học (Level 1 - Meta-learner) nhằm tối ưu hóa trọng số dự báo và giảm thiểu sai lệch tổng thể.

| Tiêu chí so sánh                 | Mô hình kinh tế lượng truyền thống (ARIMA, GARCH)                 | Học máy đơn lẻ (XGBoost, SVM, RF)                               | Học sâu đơn lẻ (LSTM, GRU, Transformer)                             | Kiến trúc Hybrid Stacking đề xuất (Tích hợp đa nền tảng)                   |
| -------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Khả năng xử lý phi tuyến         | Hạn chế, chủ yếu giả định mối quan hệ tuyến tính.                 | Tốt với dữ liệu dạng bảng nhưng thiếu tính liên kết thời gian.  | Xuất sắc trong việc nắm bắt phi tuyến tính chuỗi thời gian.         | Tối ưu hóa toàn diện bằng cách kết hợp cả quan hệ không gian và thời gian. |
| Kiểm soát phương sai & Chệch     | Độ chệch cao khi cấu trúc thị trường thay đổi đột ngột.           | Dễ bị quá khớp nếu không có cơ chế chuẩn hóa chặt chẽ.          | Phương sai cao, yêu cầu tập dữ liệu lớn để hội tụ ổn định.          | Giảm thiểu đồng thời cả độ chệch và phương sai qua meta-learning.          |
| Xử lý đặc trưng hỗn hợp          | Kém, chỉ làm việc tốt với chuỗi đơn biến hoặc đa biến tuyến tính. | Tốt với đặc trưng tĩnh và chỉ báo kỹ thuật thủ công.            | Tốt với chuỗi thời gian thô nhưng kém với dữ liệu bảng tĩnh.        | Tối ưu hóa việc tích hợp cả thuộc tính tĩnh và động lượng liên thị trường. |
| Thích ứng dịch chuyển trạng thái | Rất kém, mô hình cần được tái ước lượng liên tục.                 | Trung bình, phụ thuộc vào tần suất cập nhật dữ liệu huấn luyện. | Trung bình, dễ bị suy giảm hiệu năng khi gặp chế độ thị trường mới. | Cao, nhờ sự đa dạng hóa kiến trúc và cơ chế lọc thông minh.                |

## Khung Lý Thuyết Về Mô Hình Hybrid Stacking

### Bản Chất Toán Học Của Stacked Generalization

Khái niệm học xếp chồng (Stacked Generalization), được khởi xướng bởi Wolpert (1992) và được Breiman (1996) hình thức hóa dưới dạng lý thuyết siêu học (super learning), là một kỹ thuật học máy tích hợp sử dụng một mô hình meta để tối ưu hóa sự kết hợp của các dự đoán từ nhiều mô hình cơ sở khác nhau.

Giả sử tập dữ liệu huấn luyện ban đầu có dạng:

$$D = \{(x_i, y_i)\}_{i=1}^N$$

Trong đó $x_i \in \mathbb{R}^P$ đại diện cho véc-tơ đặc trưng đầu vào (bao gồm giá lịch sử, chỉ báo kỹ thuật, biến số vĩ mô) và $y_i \in \{-1, 0, 1\}$ đại diện cho nhãn tín hiệu giao dịch thực tế tương ứng.

Tại Tầng 0 (Level 0), một tập hợp gồm $M$ mô hình học máy cơ sở khác nhau $\{f_m\}_{m=1}^M$ được huấn luyện. Để tránh hiện tượng rò rỉ dữ liệu (data leakage) và thiên kiến quá khớp (overfitting) khi huấn luyện mô hình siêu học ở Tầng 1, quy trình kiểm định chéo phân đoạn (K-fold cross-validation) được áp dụng bắt buộc. Tập dữ liệu $D$ được chia thành $K$ phần không chồng lấn có kích thước xấp xỉ nhau.

Gọi $f_{m}^{(-k)}$ là mô hình cơ sở thứ $m$ được huấn luyện trên toàn bộ tập dữ liệu ngoại trừ phần thứ $k$. Với mỗi điểm dữ liệu $x_i$ thuộc phần thứ $k$, mô hình này sẽ tạo ra một dự báo ngoài phân đoạn (out-of-fold - OOF prediction):

$$z_{i, m} = f_{m}^{(-k)}(x_i)$$

Sau khi lặp lại quy trình này trên tất cả các phân đoạn $K$, ta thu được một tập dữ liệu mới $D'$ dành riêng cho việc huấn luyện mô hình siêu học ở Tầng 1:

$$D' = \{(z_i, y_i)\}_{i=1}^N$$

Trong đó $z_i = [z_{i, 1}, z_{i, 2}, \dots, z_{i, M}]^T \in \mathbb{R}^M$ là véc-tơ chứa các dự đoán từ $M$ mô hình cơ sở đối với quan sát thứ $i$.

Mô hình siêu học $g$ (Level 1 Meta-learner) sẽ được huấn luyện trực tiếp trên tập dữ liệu $D'$ này. Mục tiêu của mô hình siêu học là tìm ra một hàm ánh xạ tối ưu nhằm giảm thiểu sai số tổng quát hóa:

$$\hat{y}_i = g(z_i) = g\left(f_{1}(x_i), f_{2}(x_i), \dots, f_{M}(x_i)\right)$$

Trong các bài toán phân loại giao dịch tài chính, mô hình siêu học thường được lựa chọn là các cấu trúc có tính chuẩn hóa cao để tránh khuếch đại nhiễu từ tầng cơ sở, ví dụ như Hồi quy Logistic chuẩn hóa L1/L2 (Lasso/Ridge Regression). Phương thức này giúp triệt tiêu trọng số của các mô hình cơ sở hoạt động kém hiệu quả bằng cách ép hệ số tương ứng về mức không.

### Sự Kết Hợp Hỗn Hợp (Hybrid) Giữa Các Trường Phái Mô Hình

Sức mạnh tối ưu của cấu trúc Hybrid Stacking nằm ở sự đa dạng về mặt kiến trúc lý thuyết của các thuật toán Tầng 0. Đề tài này đề xuất kết hợp ba nhóm mô hình có bản chất toán học bổ trợ cho nhau bao gồm mô hình học sâu chuỗi thời gian, cây quyết định tăng cường độ dốc và mô hình khoảng cách/thống kê cổ điển.

Kiến trúc tuần hướng hai chiều (Bidirectional Long Short-Term Memory - BiLSTM và Bidirectional Gated Recurrent Unit - BiGRU) được sử dụng để trích xuất các đặc trưng động học thời gian từ cả hai hướng quá khứ và tương lai trong một cửa sổ dữ liệu xác định. Cấu trúc của một đơn vị LSTM tiêu chuẩn được định nghĩa bởi hệ thống các cổng toán học sau:

- **Cổng quên (Forget Gate):**

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

- **Cổng vào (Input Gate):**

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

- **Giá trị bộ nhớ ứng viên (Candidate Memory State):**

$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

- **Cập nhật trạng thái bộ nhớ (Cell State Update):**

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

- **Cổng ra (Output Gate):**

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

- **Trạng thái ẩn (Hidden State):**

$$h_t = o_t \odot \tanh(C_t)$$

Tương tự, mô hình GRU tinh giản cấu trúc bộ nhớ bằng cách gộp trạng thái ẩn và trạng thái bộ nhớ, được kiểm soát thông qua hai cổng chính:

- **Cổng thiết lập lại (Reset Gate):**

$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

- **Cổng cập nhật (Update Gate):**

$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

- **Trạng thái ẩn ứng viên (Candidate Hidden State):**

$$\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t] + b_h)$$

- **Trạng thái ẩn cuối cùng (Final Output):**

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

Sự kết hợp của các cấu trúc tuần tuần hoàn này với mạng Transformer tích hợp cơ chế tự chú ý (Self-Attention) cho phép hệ thống nhận diện các phụ thuộc xa hơn về mặt thời gian và các cú sốc kinh tế có độ trễ lớn.

Bên cạnh đó, các thuật toán dựa trên cây quyết định như XGBoost, LightGBM và CatBoost đóng vai trò tối ưu hóa việc phân tách phi tuyến tính trên dữ liệu dạng bảng tĩnh và các chỉ báo kỹ thuật. XGBoost hoạt động dựa trên nguyên lý tối ưu hóa hàm mục tiêu được xấp xỉ Taylor bậc hai để thêm các cây quyết định mới sửa chữa sai số của các cây trước đó. LightGBM tối ưu hóa tốc độ và không gian nhớ nhờ kỹ thuật Exclusive Feature Bundling (EFB) và thuật toán phân tách dựa trên biểu đồ tần suất (Histogram-based). CatBoost giải quyết triệt để vấn đề dịch chuyển hiệp biến (covariate shift) và rò rỉ mục tiêu nhờ phương pháp hoán vị ngẫu nhiên trong quá trình tính toán giá trị phân loại. Khi kết hợp hai trường phái này, mô hình siêu học (Meta-learner) sử dụng thuật toán Hồi quy Logistic có dạng:

$$p(X; b, w) = \frac{1}{1 + e^{-(w \cdot X + b)}}$$

Phương trình này ánh xạ tổng các đầu ra từ tầng cơ sở thành một phân phối xác suất phân loại chuẩn hóa, cho phép đưa ra ranh giới quyết định giao dịch rõ ràng.

| Thuật toán                | Vai trò kiến trúc                                | Siêu tham số tối ưu hóa chính                                         | Mục tiêu toán học trong huấn luyện                                   |
| ------------------------- | ------------------------------------------------ | --------------------------------------------------------------------- | -------------------------------------------------------------------- |
| BiLSTM                    | Bộ trích xuất động lực chuỗi thời gian (Tầng 0). | hidden_units (64, 128, 256), dropout_rate (0.2 - 0.5), learning_rate. | Tối thiểu hóa sai số entropy chéo nhị phân hoặc đa lớp.              |
| XGBoost                   | Bộ khai thác phi tuyến cấu trúc tĩnh (Tầng 0).   | max_depth (3 - 10), eta (0.01 - 0.1), subsample (0.6 - 0.8).          | Tối ưu hóa hàm mất mát chính quy hóa bậc hai.                        |
| LightGBM                  | Bộ tính toán đặc trưng tần suất cao (Tầng 0).    | num_leaves (31 - 127), min_data_in_leaf, feature_fraction.            | Tăng tốc độ hội tụ thông qua kỹ thuật phân tách theo lá (Leaf-wise). |
| Support Vector Classifier | Bộ phân tách biên tối đa (Tầng 0).               | C (độ phạt điều hòa), kernel (RBF hoặc Tuyến tính), gamma.            | Tối đa hóa khoảng cách lồi giữa các biên quyết định phân lớp.        |
| Logistic Regression       | Bộ siêu học hội tụ (Tầng 1 Meta-learner).        | penalty ('l1' hoặc 'l2'), C (nghịch đảo của hệ số chuẩn hóa).         | Tối ưu hóa xác suất hợp lý cực đại (Maximum Likelihood Estimation).  |

## Kỹ Thuật Tiền Xử Lý Dữ Liệu Và Trích Xuất Đặc Trưng Nâng Cao

### Kỹ Thuật Sai Phân Phân Số (Fractional Differencing)

Một trong những thách thức nghiêm trọng nhất khi xử lý dữ liệu tài chính như giá vàng là sự mâu thuẫn giữa tính dừng (stationarity) và tính lưu giữ thông tin (memory retention). Sai phân bậc một ($d=1$) tạo ra tỷ suất sinh lợi stationary, điều này là bắt buộc đối với hầu hết các thuật toán học máy. Tuy nhiên, phép biến đổi này đã triệt tiêu hoàn toàn thông tin về mức giá lịch sử, làm mất đi các sự kiện mang tính chu kỳ dài hạn và "ký ức" của thị trường.

Để giải quyết vấn đề này, đề tài ứng dụng lý thuyết Sai phân Phân số được giới thiệu bởi Marcos López de Prado. Phép toán sai phân phân số với bậc $d \in \mathbb{R}$ được định nghĩa thông qua chuỗi khai triển nhị thức Newton vô hạn của toán tử trễ $B$:

$$(1-B)^d = \sum_{k=0}^{\infty} (-1)^k \binom{d}{k} B^k = \sum_{k=0}^{\infty} \omega_k B^k$$

Trong đó các trọng số $\omega_k$ được tính toán theo công thức truy hồi dưới đây:

$$\omega_k = \omega_{k-1} \frac{d - k + 1}{k}, \quad \text{với } \omega_0 = 1$$

Khi áp dụng vào chuỗi giá vàng $X_t$, giá trị sai phân phân số tại thời điểm $t$ là:

$$\tilde{X}_t = \sum_{k=0}^{l} \omega_k X_{t-k}$$

Trong thực tế triển khai, chuỗi vô hạn được cắt giảm tại một ngưỡng dung sai $\tau = 10^{-4}$ nhằm cân bằng giữa tính hiệu quả tính toán và việc bảo toàn ký ức toán học dài hạn. Quy trình thực nghiệm yêu cầu quét giá trị $d$ từ $0$ đến $1$ với bước nhảy $0.05$. Bậc sai phân tối ưu $d^*$ được xác định là giá trị nhỏ nhất mà tại đó chuỗi $\tilde{X}_t$ vượt qua bài kiểm tra tính dừng Augmented Dickey-Fuller (ADF) với mức ý nghĩa $1\%$ ($p\text{-value} < 0.01$). Phương pháp này giúp mô hình Hybrid Stacking tiếp cận được nguồn dữ liệu vừa stationary vừa lưu trữ tối đa thông tin cấu trúc giá ban đầu.

### Biến Đổi Sóng Nhỏ (Wavelet Transform) Để Khử Nhiễu

Giá vàng CFD chứa đựng lượng nhiễu trắng cực kỳ lớn phát sinh từ các giao dịch tần suất cao. Việc đưa trực tiếp dữ liệu nhiễu vào các mô hình học sâu chuỗi thời gian thường dẫn đến việc học các mẫu giả ngẫu nhiên. Đề tài đề xuất sử dụng Biến đổi sóng nhỏ rời rạc (Discrete Wavelet Transform - DWT) dựa trên phân tích đa độ phân giải (Multi-Resolution Analysis - MRA) nhằm bóc tách tín hiệu thực tế ra khỏi nhiễu thị trường.

Khác với biến biến đổi Fourier truyền thống chỉ hoạt động trên miền tần số và làm mất thông tin thời gian, Wavelet Transform cung cấp khả năng định vị đồng thời trên cả miền thời gian lẫn tần số bằng cách dịch chuyển và co giãn một hàm sóng mẹ $\psi(t)$. Công thức biến đổi wavelet liên tục là:

$$(Wf)(s, b) = \frac{1}{\sqrt{s}} \int_{-\infty}^{\infty} f(t) \psi^* \left( \frac{t - b}{s} \right) dt$$

Trong đó $s$ là tham số tỷ lệ (scale) biểu thị cho tần số và $b$ là tham số dịch chuyển biểu thị cho thời gian. Trong môi trường rời rạc, chuỗi giá vàng được phân rã thành hai nhóm hệ số ở mỗi cấp độ phân rã $j$:

- **Các hệ số xấp xỉ (Approximation Coefficients - $a_j$):** Đại diện cho các xu hướng tần số thấp, dài hạn của giá vàng.

- **Các hệ số chi tiết (Detail Coefficients - $d_j$):** Đại diện cho các biến động tần số cao, ngắn hạn và nhiễu.

Quy trình khử nhiễu sóng nhỏ bao gồm: phân rã chuỗi giá vàng đến cấp độ tối ưu $J$ (thường từ $3$ đến $5$), áp dụng ngưỡng mềm (soft thresholding) được tối ưu hóa theo quy tắc Stein's Unbiased Risk Estimate (SURE) lên các hệ số chi tiết $d_j$ để loại bỏ nhiễu, và cuối cùng thực hiện biến đổi sóng nhỏ ngược (IDWT) để tái cấu trúc chuỗi giá sạch. Các họ wavelet như Coiflets (đặc biệt là Coif5 do tính đối xứng cao và khả năng bảo toàn đặc trưng biên) hoặc Daubechies (db4) được khuyên dùng cho chuỗi thời gian tài chính.

### Kỹ Thuật Điền Khuyết Và Chuẩn Hóa

Dữ liệu lịch sử thu thập từ sàn giao dịch hoặc các nguồn thông tin kinh tế thường xuất hiện tình trạng mất mát dữ liệu do lệch múi giờ giao dịch hoặc ngày nghỉ lễ. Để bảo toàn cấu trúc dữ liệu, thuật toán K-Nearest Neighbors (kNN) được áp dụng để điền khuyết. Khoảng cách Minkowski giữa hai quan sát $x$ và $y$ được định nghĩa là:

$$d(x, y) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{1/p}$$

Với $p = 2$ tương đương khoảng cách Euclidean, giá trị khuyết thiếu được ước tính bằng trung bình có trọng số của $K$ láng giềng gần nhất. Tiếp theo, các biến số liên tục được chuẩn hóa Min-Max về đoạn $[0,1]$ nhằm ổn định quá trình lan truyền ngược trong các mạng nơ-ron:

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

### Thiết Kế Tập Tính Năng Toàn Diện (Feature Engineering Suite)

Để cung cấp một góc nhìn đa chiều cho mô hình Hybrid Stacking, tập đặc trưng đầu vào được thiết kế bao gồm ba nhóm thuộc tính chính và được biểu diễn chi tiết qua hệ thống công thức toán học dưới đây:

| Tên chỉ báo | Công thức toán học xác định | Mô tả ý nghĩa tài chính |
| --- | --- | --- |
| Exponential Moving Average (EMA) | $EMA_t = \left(Close_t \times \frac{2}{N+1}\right) + \left(EMA_{t-1} \times \left(1 - \frac{2}{N+1}\right)\right)$ | Xác định hướng đi của xu hướng hiện tại bằng cách gán trọng số cao hơn cho các mức giá gần nhất. |
| Moving Average Convergence Divergence (MACD) | $MACD = EMA_{12}(Close) - EMA_{26}(Close)$ | Đo lường gia tốc xu hướng và phát hiện các điểm đảo chiều thông qua sự hội tụ/phân kỳ của hai đường trung bình động. |
| Relative Strength Index (RSI) | $RSI = 100 - \frac{100}{1 + \frac{\text{Average Gain}}{\text{Average Loss}}}$ | Đánh giá cường độ biến động giá để nhận biết các vùng thị trường quá mua hoặc quá bán. |
| Stochastic Oscillator (SO) | $SO = \frac{Close_t - Low_N}{High_N - Low_N} \times 100$ | So sánh giá đóng cửa hiện tại với phạm vi giá trong một khoảng thời gian $N$ để dự báo điểm xoay chiều. |
| Awesome Oscillator (AO) | $AO = SMA_5\left(\frac{High+Low}{2}\right) - SMA_{34}\left(\frac{High+Low}{2}\right)$ | Đo lường động lượng thị trường bằng cách so sánh các chu kỳ ngắn hạn và dài hạn. |
| Average True Range (ATR) | $TR_t = \max(High_t - Low_t, \lvert High_t - Close_{t-1} \rvert, \lvert Low_t - Close_{t-1} \rvert)$ | Đo lường biên độ dao động thực tế của thị trường, hỗ trợ thiết lập stop loss và take profit động. |
| Bollinger Bands (BB) | $Mid = SMA_{20}(Close);\quad Band = Mid \pm k \times \sigma_{20}(Close)$ | Thiết lập các ranh giới biến động động, hỗ trợ xác định vùng đột phá giá hoặc củng cố xu hướng. |

## Thiết Kế Nhãn Mục Tiêu Và Kiểm Định Chéo Khử Chồng Lấn

### Thiết Kế Nhãn Mục Tiêu Bằng Phương Pháp Ba Rào Chắn (Triple-Barrier Method)

Việc sử dụng phương pháp gắn nhãn truyền thống dựa trên dấu của tỷ suất sinh lợi cố định sau một khoảng thời gian ($y_t = \text{sign}(P_{t+h} - P_t)$) thường gây ra sai số lớn trong thực tế giao dịch. Phương pháp này bỏ qua các biến động giá cực đại xảy ra bên trong khoảng thời gian dự báo, dẫn đến việc mô hình kích hoạt tín hiệu mua nhưng thực tế tài khoản đã bị cắt lỗ (margin call) trước khi đạt tới thời điểm $t+h$.

Đề tài giải quyết vấn đề này bằng việc triển khai Phương pháp Ba rào chắn (Triple-Barrier Method) kết hợp kỹ thuật Meta-Labeling của López de Prado. Phương pháp này thiết lập ba rào cản động xung quanh điểm vào lệnh tại thời điểm $t$:

- **Rào chắn phía trên (Upper Barrier):** Đại diện cho mục tiêu chốt lời (Take Profit - TP), được xác định động bằng mức giá đầu vào cộng thêm một bội số của độ biến động ATR hiện tại: $P_t + \beta \times \text{ATR}_t$.

- **Rào chắn phía dưới (Lower Barrier):** Đại diện cho mức dừng lỗ (Stop Loss - SL), được xác định động bằng: $P_t - \alpha \times \text{ATR}_t$.

- **Rào chắn thời gian (Horizontal Barrier):** Giới hạn thời gian nắm giữ vị thế tối đa gồm $T$ nến giao dịch.

Nhãn mục tiêu sơ cấp $y_t \in \{-1, 0, 1\}$ được gán như sau:

$$
y_t = \begin{cases}
1 & \text{nếu đường giá chạm rào chắn phía trên trước (Tín hiệu Mua)} \\
-1 & \text{nếu đường giá chạm rào chắn phía dưới trước (Tín hiệu Bán)} \\
0 & \text{nếu đường giá không chạm cả hai rào chắn và hết thời gian } T \text{ (Đứng ngoài)}
\end{cases}
$$

Kỹ thuật Meta-Labeling sau đó được áp dụng bằng cách huấn luyện một mô hình nhị phân thứ cấp để dự báo xem mô hình Hybrid Stacking sơ cấp có đưa ra quyết định đúng hay không (nhãn nhị phân $y^{meta}_t \in \{0, 1\}$), từ đó tối ưu hóa việc phân bổ quy mô vị thế giao dịch (position sizing) và giảm thiểu rủi ro thua lỗ ròng.

### Kiểm Định Chéo Loại Bỏ Chồng Lấn (Purged and Embargoed Cross-Validation)

Trong dữ liệu tài chính, các nhãn mục tiêu thường chồng chéo về mặt thời gian (ví dụ: một vị thế mở tại thời điểm $t$ kéo dài đến $t+4$ mới chạm rào chắn, trùng lặp với vị thế mở tại $t+1$). Nếu sử dụng phương pháp chia K-fold thông thường, thông tin từ tương lai trong tập kiểm tra sẽ rò rỉ vào tập huấn luyện thông qua các thuộc tính trùng lặp, tạo ra kết quả kiểm thử cực kỳ lạc quan nhưng thất bại thảm hại khi giao dịch thực tế. Quy trình huấn luyện đề tài bắt buộc phải tích hợp hai cơ chế bảo vệ nghiêm ngặt:

- **Purging (Loại bỏ chồng lấn):** Bất kỳ quan sát nào trong tập huấn luyện có khoảng thời gian nhãn mục tiêu chồng lấn với khoảng thời gian của tập kiểm định đều bị loại bỏ hoàn toàn khỏi quá trình huấn luyện. Điều này đảm bảo rằng mô hình không được học từ bất kỳ thông tin nào phát sinh trong thời gian đánh giá của tập kiểm thử.

- **Embargoing (Cấm vận tạm thời):** Do tính tự tương quan (autocorrelation) của các biến số tài chính, thông tin tại điểm kết thúc của tập kiểm định vẫn có thể ảnh hưởng đến các điểm dữ liệu ngay sau đó trong tập huấn luyện. Một khoảng thời gian cấm vận (thường chiếm từ $1\%$ đến $5\%$ tổng chiều dài chuỗi dữ liệu) được áp dụng ngay sau tập kiểm định. Toàn bộ các mẫu huấn luyện rơi vào vùng cấm vận này sẽ bị xóa bỏ.

Đề tài triển khai cấu trúc Kiểm định chéo loại bỏ chồng lấn tổ hợp (Combinatorial Purged Cross-Validation - CPCV). CPCV chia tập dữ liệu thành $N$ nhóm tuần tự, tạo ra các tổ hợp huấn luyện-kiểm tra phức tạp giúp xây dựng nhiều đường kiểm thử lịch sử (backtest paths) khác nhau. Điều này cho phép tạo ra một phân phối thực tế của hệ số Sharpe và các chỉ số hiệu năng khác thay vì chỉ một điểm ước lượng duy nhất, tăng cường khả năng phát hiện hiện tượng quá khớp kiểm thử (backtest overfitting).

| Thuộc tính so sánh | K-Fold Cross Validation thông thường | Walk-Forward Backtesting truyền thống | Combinatorial Purged Cross Validation (CPCV) đề xuất |
| --- | --- | --- | --- |
| Giả định phân phối dữ liệu | Độc lập và đồng nhất (I.I.D) - Không đúng với tài chính. | Chuỗi thời gian phi dừng, phụ thuộc tuyến tính. | Chuỗi thời gian phi dừng có tính đến chồng lấn thông tin nhãn. |
| Kiểm soát rò rỉ thông tin | Hoàn toàn không có, dẫn đến kết quả ảo tưởng. | Có kiểm soát theo chiều thời gian tiến tới nhưng hiệu quả sử dụng dữ liệu thấp. | Triệt để bằng cơ chế Purging và Embargo tự động hóa. |
| Độ phủ dữ liệu kiểm thử | Toàn bộ tập dữ liệu được kiểm tra một lần. | Chỉ kiểm tra phần dữ liệu lịch sử muộn nhất. | Tạo ra nhiều đường kiểm thử phân tán trên toàn bộ chuỗi lịch sử. |
| Khả năng ước lượng rủi ro | Kém, dễ bị ảnh hưởng bởi nhiễu cục bộ. | Phụ thuộc mạnh vào chu kỳ thị trường được chọn để kiểm thử. | Cung cấp phân phối xác suất đầy đủ của các chỉ số hiệu năng. |

## Quy Trình Triển Khai Thực Nghiệm Hệ Thống Giao Dịch

### Thiết Kế Quy Trình Thực Nghiệm Tuần Tự

Quy trình thực hiện đồ án được thiết kế thành một lộ trình hệ thống, đi từ khâu hạ tầng dữ liệu đến tối ưu hóa mô hình và kiểm thử thực tế. Quy trình này được mô phỏng chi tiết thông qua bảng phân rã các giai đoạn triển khai thực nghiệm dưới đây:

| Giai đoạn triển khai              | Nhiệm vụ kỹ thuật chi tiết                                                                                                   | Công cụ & thư viện sử dụng                 | Sản phẩm đầu ra kỳ vọng                                                 |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ | ----------------------------------------------------------------------- |
| Giai đoạn 1: Thu thập & Làm sạch  | Tải dữ liệu XAU/USD H1, chỉ số vĩ mô (DXY, US10Y), giá hàng hóa liên quan. Điền khuyết kNN và chuẩn hóa dữ liệu.             | Python, yfinance, pandas, scikit-learn.    | Cơ sở dữ liệu chuỗi thời gian sạch, định dạng đồng bộ thời gian.        |
| Giai đoạn 2: Tiền xử lý nâng cao  | Thực thi toán tử sai phân phân số với bậc tối ưu $d^*$ qua kiểm định ADF. Phân rã đa độ phân giải DWT Coif5 khử nhiễu.       | statsmodels (ADF test), PyWavelets (DWT).  | Chuỗi dữ liệu stationary được triệt tiêu nhiễu tần số cao.              |
| Giai đoạn 3: Tính toán & Gắn nhãn | Tính toán hệ thống chỉ báo kỹ thuật. Khởi tạo rào chắn động Triple-Barrier dựa trên ATR để gán nhãn mục tiêu.                | TA-Lib, numpy, pandas.                     | Tập dữ liệu hoàn chỉnh chứa đặc trưng đầu vào và nhãn mục tiêu thực tế. |
| Giai đoạn 4: Tích hợp hệ thống    | Thiết lập cấu trúc Stacking hai tầng. Tầng 0 huấn luyện BiLSTM, XGBoost, LightGBM, SVC. Tầng 1 sử dụng Logistic Regression.  | PyTorch (BiLSTM/BiGRU), xgboost, lightgbm. | Kiến trúc mô hình Hybrid Stacking hoàn chỉnh.                           |
| Giai đoạn 5: Tối ưu hóa & Lọc     | Áp dụng GA-PSO tối ưu hóa đồng thời siêu tham số và trọng số xếp chồng. Thực hiện cơ chế lọc thông minh loại bỏ mô hình yếu. | DEAP (Genetic Algorithm), scipy.optimize.  | Phiên bản mô hình tối ưu hóa biên lợi nhuận và độ chính xác.            |
| Giai đoạn 6: Mô phỏng giao dịch   | Thực thi kiểm định chéo CPCV. Mô phỏng giao dịch tích hợp spread, swap, slippage, đòn bẩy tài chính.                         | Backtrader, pyalgotrade.                   | Báo cáo chi tiết hiệu năng tài chính và các chỉ số quản trị rủi ro.     |

### Các Giải Thuật Trích Chọn Đặc Trưng Và Tối Ưu Hóa Siêu Tham Số

Để nâng cao hiệu quả tính toán và loại bỏ hiện tượng đa cộng tuyến (multicollinearity) giữa các chỉ báo kỹ thuật, đồ án áp dụng tích hợp các giải thuật trích chọn đặc trưng tiên tiến bao gồm:

- **Phân tích thành phần chính (Principal Component Analysis - PCA):** Giảm chiều dữ liệu bằng cách chiếu không gian đặc trưng ban đầu lên các trục trực giao có phương sai lớn nhất, giúp loại bỏ nhiễu tuyến tính.

- **Giải thuật lựa chọn đặc trưng dựa trên thông tin tương hỗ (Feature Selection based on Mutual Information - FCMIM):** Đo lường mức độ phụ thuộc phi tuyến tính giữa các thuộc tính đầu vào và nhãn mục tiêu để giữ lại các biến có khả năng giải thích cao nhất.

- **Giải thuật ReliefF kết hợp Giải thuật Di truyền (ReliefF-GA):** ReliefF đánh giá tầm quan trọng của các thuộc tính dựa trên khoảng cách giữa các cặp quan sát gần nhất thuộc cùng một lớp và khác lớp, sau đó Giải thuật Di truyền tìm kiếm không gian tổ hợp các đặc trưng tối ưu để tối đa hóa hiệu suất phân loại.

Về khía cạnh tối ưu hóa siêu tham số (Hyperparameter Optimization - HPO), đồ án triển khai phương pháp tối ưu hóa đồng thời toàn diện (Stacked_Hybrid_Full) thay vì tối ưu hóa đơn lẻ từng thành phần. Sử dụng thuật toán lai kết hợp Giải thuật Di truyền và Tối ưu hóa bầy đàn (GA-PSO) để tìm kiếm đồng thời:

- Tập hợp các siêu tham số tối ưu $\theta_m$ cho từng mô hình cơ sở ở Tầng 0.
- Các trọng số xếp chồng (stacking weights) $\beta$ tối ưu của mô hình siêu học ở Tầng 1.

Mục tiêu của quá trình tối ưu hóa này là tối thiểu hóa hàm mất mát trên tập kiểm định ngoài phân đoạn (OOF loss):

$$\min_{\theta, \beta} \mathcal{L}_{OOF} = \sum_{i=1}^N \mathcal{L}\left( y_i, g\left( \sum_{m=1}^M \beta_m f_m(x_i; \theta_m) \right) \right)$$

### Cơ Chế Lọc Mô Hình Và Sự Đồng Thuận Tín Hiệu (Smart Filtering)

Đồ án tích hợp một cơ chế kiểm soát chất lượng nghiêm ngặt dựa trên nguyên lý "Chất lượng hơn Số lượng". Thực nghiệm chứng minh rằng việc kết hợp bừa bãi toàn bộ các mô hình cơ sở thường làm suy giảm hiệu năng của hệ thống xếp chồng do tác động của các mô hình yếu. Quy trình lọc bao gồm hai bước chính:

- **Lọc chất lượng sơ cấp (Smart Filtering):** Tự động loại bỏ bất kỳ mô hình cơ sở nào có độ chính xác dự báo hướng đi thấp hơn ngưỡng $52\%$ trên dữ liệu kiểm thử OOF trước khi tiến hành tổng hợp Stacking.

- **Bộ lọc đồng thuận dựa trên độ tin cậy (Confidence-based Filtering):** Hệ thống chỉ kích hoạt và thực thi lệnh giao dịch thực tế khi có sự đồng thuận tín hiệu từ một số lượng mô hình cơ sở tối thiểu (ví dụ: tối thiểu $6$ trong số $9$ mô hình cơ sở cùng đưa ra một hướng tín hiệu Buy hoặc Sell). Phương pháp này giúp nâng cao đáng kể hệ số Sharpe và giảm thiểu các giao dịch sai lệch do nhiễu thị trường cục bộ.

### Tích Hợp Ràng Buộc Tài Chính Thực Tế (Post-hoc Finance-Informed Refinement)

Một đóng góp đổi mới mang tính thực tiễn cao của đề tài là việc áp dụng bước Tinh chỉnh Tài chính sau mô hình (Post-hoc Finance-Informed Refinement). Thay vì hoàn toàn tin tưởng vào các dự báo thuần toán học của hệ thống học máy (vốn có thể đưa ra các tín hiệu phi lý trong các giai đoạn thị trường cực đoan), đồ án đề xuất blending dự báo của mô hình Stacking với các tiên nghiệm kinh tế (economic priors) dưới dạng các ràng buộc mềm (soft regularizers).

Giả sử dự báo thô của mô hình siêu học cho xác suất tăng giá là $\hat{y}_i = s_i \beta$. Ta tính toán một giá trị tiên nghiệm tài chính $y^{prior}_i$ dựa trên các mô hình cân bằng hoặc lý thuyết biên độ lịch sử (ví dụ: tỷ lệ Gold-to-Silver Ratio vượt quá độ lệch chuẩn $2\sigma$ biểu thị xu hướng đảo chiều trung bình mạnh, hoặc chênh lệch lãi suất thực tế Mỹ đạt mức cực đại). Tín hiệu giao dịch cuối cùng được hiệu chỉnh thông qua phép tổ hợp lồi:

$$\hat{y}^{final}_i = (1 - \lambda) \hat{y}_i + \lambda y^{prior}_i$$

Trong đó $\lambda \in [0, 0.3]$ là hệ số điều phối ràng buộc vĩ mô. Phương pháp này đảm bảo hệ thống giao dịch duy trì tính nhất quán lý thuyết tài chính và bảo vệ tài khoản khỏi các biến động "Thiên nga đen" mà mô hình học máy chưa từng được trải nghiệm trong dữ liệu huấn luyện lịch sử.

## Hệ Thống Chỉ Số Đánh Giá Và Mô Phỏng Chiến Lược

### Đo Lường Hiệu Năng Toàn Diện

Sự thành bại của một hệ thống giao dịch CFD vàng không thể được đánh giá phiến diện qua các chỉ số học máy như sai số MSE hay độ chính xác hướng đi. Đồ án xây dựng một ma trận chỉ số đánh giá đa chiều, liên kết chặt chẽ giữa hiệu suất kỹ thuật và kết quả tài chính thực tế.

| Nhóm chỉ số | Tên chỉ số | Công thức toán học / mô tả | Ý nghĩa thực tế trong giao dịch vàng |
| --- | --- | --- | --- |
| Hiệu năng học máy (ML Metrics) | Precision (Độ chính xác) | $P = \frac{TP}{TP + FP}$ | Tỷ lệ tín hiệu giao dịch chính xác trên tổng số lệnh được phát ra, quyết định trực tiếp đến chi phí giao dịch ròng. |
| Hiệu năng học máy (ML Metrics) | Recall (Độ nhạy) | $R = \frac{TP}{TP + FN}$ | Khả năng của mô hình trong việc nắm bắt các xu hướng lớn của giá vàng. |
| Hiệu năng học máy (ML Metrics) | ROC-AUC | Diện tích dưới đường cong ROC | Đo lường năng lực phân biệt trạng thái thị trường tăng/giảm của mô hình xếp chồng. |
| Hiệu năng tài chính (Trading Metrics) | Sharpe Ratio ($SR$) | $SR = \frac{E}{\sigma_p}$ | Tỷ suất sinh lợi vượt trội thu được trên mỗi đơn vị rủi ro chấp nhận, tối ưu hóa hiệu quả sử dụng vốn. |
| Hiệu năng tài chính (Trading Metrics) | Calmar Ratio ($CR$) | $CR = \frac{\text{Annualized Return}}{\text{Maximum Drawdown}}$ | Đánh giá khả năng phục hồi tài sản của chiến lược sau các giai đoạn sụt giảm nghiêm trọng. |
| Hiệu năng tài chính (Trading Metrics) | Maximum Drawdown ($MDD$) | $MDD = \max \left( \frac{Peak_t - Valley_t}{Peak_t} \right)$ | Thước đo rủi ro sụt giảm vốn lớn nhất, quyết định giới hạn đòn bẩy an toàn cho tài khoản. |
| Hiệu năng tài chính (Trading Metrics) | Profit Factor ($PF$) | $PF = \frac{\sum \text{Lợi nhuận từ lệnh thắng}}{\sum \text{Thua lỗ từ lệnh thua}}$ | Chỉ số đo lường tính hiệu quả kinh tế tổng thể của hệ thống giao dịch. |

Mối quan hệ nhân quả ở đây thể hiện ở việc: một mô hình có độ chính xác (Precision) cao ở Tầng 1 sẽ trực tiếp làm giảm tỷ lệ tín hiệu giả (False Positives), từ đó hạn chế số lượng giao dịch thua lỗ không đáng có, trực tiếp bảo vệ tài khoản khỏi các cú sụt giảm sâu (Maximum Drawdown thấp) và gián tiếp tối ưu hóa Sharpe Ratio của toàn bộ chiến lược.

### Giải Thích Mô Hình Bằng Lý Thuyết Trò Chơi (SHAP)

Nhằm phá bỏ rào cản "hộp đen" của các kiến trúc tích hợp phức tạp, đồ án áp dụng phương pháp giải thích SHAP (SHapley Additive exPlanations) dựa trên nền tảng lý thuyết trò chơi hợp tác. SHAP phân rã đóng góp của từng đặc trưng đầu vào đối với đầu ra dự báo xác suất tín hiệu giao dịch.

Giá trị SHAP $\phi_i(x)$ của đặc trưng thứ $i$ được tính bằng:

$$
\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[f_{S \cup \{i\}}(x_{S \cup \{i\}}) - f_S(x_S)\right]
$$

Thông qua việc trực quan hóa biểu đồ lực lượng SHAP (SHAP Summary Plot), nhà nghiên cứu có thể xác định chính xác các mối liên hệ nhân quả:

- Mức độ ảnh hưởng tương đối của các chỉ số động lượng kỹ thuật (như RSI hay MACD) so với các biến số kinh tế vĩ mô (như sức mạnh đồng USD - DXY) trong việc hình thành tín hiệu Buy/Sell.

- Sự tương tác phi tuyến tính giữa các thuộc tính (ví dụ: khi độ biến động ATR tăng cao, tầm quan trọng của các đường trung bình động EMA giảm đi đáng kể do thị trường chuyển dịch sang trạng thái dao động mạnh không xu hướng).

- Khả năng kiểm soát rủi ro hệ thống thông qua việc phát hiện các tín hiệu dự báo sai lệch phát sinh từ sự nhiễu loạn của một nhóm đặc trưng cụ thể, đáp ứng các tiêu chuẩn khắt khe về tính minh bạch trong tài chính định lượng.

### Mô Phỏng Backtesting Tích Hợp Chi Phí Ma Sát

Quá trình kiểm thử lịch sử (backtesting) bắt buộc phải tích hợp các yếu tố chi phí ma sát của thị trường CFD thực tế để tránh hiện tượng kết quả ảo tưởng. Mô hình mô phỏng cần thiết lập các tham số động bao gồm:

- **Chênh lệch Giá Mua - Bán (Spread):** CFD vàng là sản phẩm có spread động, thường giãn rộng rất mạnh trong các khung giờ giao thoa phiên (London/New York overlap) hoặc tại các thời điểm công bố tin tức kinh tế quan trọng của Mỹ.

- **Trượt giá thực thi (Slippage):** Sự sai lệch giữa giá kích hoạt tín hiệu trên lý thuyết và giá khớp lệnh thực tế do độ trễ truyền tải tín hiệu kết nối API.

- **Phí qua đêm (Swap):** Chi phí nắm giữ vị thế mua/bán qua ngày, có thể ảnh hưởng lớn đến lợi nhuận ròng của các chiến lược giao dịch trung hạn.

- **Yêu cầu ký quỹ và Đòn bẩy (Leverage & Margin):** Mô phỏng tỷ lệ đòn bẩy thực tế để tính toán chính xác điểm kích hoạt dừng giao dịch bắt buộc (Margin Call) khi tài khoản rơi vào trạng thái sụt giảm sâu.

## Kết Luận Và Định Hướng Phát Triển

### Tổng Kết Đóng Góp Công Nghệ Của Đề Tài

Đồ án "Ứng dụng mô hình Hybrid Stacking dự báo tín hiệu giao dịch CFD vàng" thiết lập một giải pháp toàn diện và khoa học cho bài toán giao dịch định lượng tài sản biến động mạnh. Bằng cách kết hợp hài hòa giữa toán học tài chính hiện đại và khoa học dữ liệu, nghiên cứu giải quyết triệt để các hạn chế cố hữu của các mô hình truyền thống. Đóng góp cốt lõi của đề tài nằm ở việc cấu trúc hóa một quy trình khép kín, ngăn ngừa hiệu quả hiện tượng rò rỉ dữ liệu thông qua CPCV, nâng cao chất lượng tín hiệu nhờ kỹ thuật khử nhiễu sóng nhỏ Wavelet và duy trì tính dừng thông qua sai phân phân số. Việc áp dụng cơ chế lọc mô hình yếu và tích hợp các ràng buộc kinh tế thực tế biến hệ thống Stacking từ một công cụ thống kê lý thuyết thành một giải pháp thực chiến có độ ổn định và an toàn vốn cao.

### Định Hướng Phát Triển Hệ Thống Trong Tương Lai

Trong các giai đoạn phát triển tiếp theo, hệ thống giao dịch định lượng CFD vàng có thể được nâng cấp thông qua hai hướng nghiên cứu mang tính tiên phong:

- **Tích hợp Mô hình Ngôn ngữ Lớn đa tác tử (Multi-Agent LLMs):** Áp dụng các mô hình ngôn ngữ lớn để tự động quét, phân tích tâm lý tin tức kinh tế vĩ mô toàn cầu, báo cáo của Cục Dự trữ Liên bang Mỹ (FED) và các bất ổn địa chính trị theo thời gian thực. Điểm số tâm lý (sentiment scores) này sẽ được mã hóa và truyền trực tiếp làm đặc trưng đầu vào bổ trợ cho tầng cơ sở của Stacking, giúp hệ thống phản ứng nhanh nhạy trước các sự kiện tin tức đột ngột.

- **Học máy lượng tử lai (Hybrid Classical-Quantum Machine Learning):** Nghiên cứu ứng dụng các hạt nhân lượng tử thông qua thuật toán Quantum Support Vector Regressor (QSVR). Việc ánh xạ dữ liệu tài chính vào không gian Hilbert lượng tử cho phép mô hình phát hiện ra các mối tương quan vướng mắc phức tạp (entangled relationships) giữa giá vàng và các nhóm tài sản liên thị trường mà các máy tính cổ điển hoàn toàn bỏ sót, mở ra triển vọng vượt trội về mặt hiệu năng dự báo tín hiệu trong các chu kỳ kinh tế mới.
