# Hedge-fund — Time-series Forecasting 

## Tóm tắt
Dự án này xây dựng một pipeline dự báo chuỗi thời gian cho bài toán “hedge fund prediction” (dạng dữ liệu bảng theo thời gian, có trọng số quan sát). Mục tiêu là dự đoán biến mục tiêu `y_target` dựa trên:

- Các đặc trưng dạng số (các cột `feature_*`),
- Các biến phân loại: `code`, `sub_code`, `sub_category`, `horizon`,
- Chỉ số thời gian `ts_index`,
- Trọng số `weight'.

Trong notebook `main.ipynb`, mô hình chính là **Transformer**, kèm theo một mô hình **LightGBM** đơn giản.

> Ghi chú: đây là dự án nghiên cứu/học tập.

---

## 1. Mô tả dữ liệu & bài toán
### 1.1. Cấu trúc dữ liệu
Các notebook sử dụng dữ liệu dạng parquet:

- `train.parquet`: gồm `id`, `ts_index`, các `feature_*`, biến phân loại (`code`, `sub_code`, `sub_category`, `horizon`), và nhãn `y_target` kèm `weight`.
- `test.parquet`: cấu trúc tương tự nhưng không có `y_target`.

### 1.2. Mục tiêu dự đoán
Cho từng dữ liệu theo thời gian, dự đoán:
$$
\hat{y} = f(\text{features}, \text{categorical context}, \text{history})
$$

---

## 2. Tiền xử lý dữ liệu

### 2.1. Xử lý thiếu dữ liệu (missing values)
Trong `Data preprocessing.ipynb`, số lượng đặc trưng được thống kê và chia thành 2 nhóm theo tỷ lệ dữ liệu rỗng:

- Nhóm rỗng ít: `null_ratio < 0.05` → điền **median**.
- Nhóm rỗng nhiều: `null_ratio >= 0.05` → tạo thêm missing indicator và vẫn điền bằng **median**.


### 2.2. Chuẩn hoá (scaling)
Trong `main.ipynb`, các đặc trưng số được chuẩn hoá bằng `StandardScaler` (fit trên train, transform trên test) để ổn định quá trình tối ưu trong mô hình deep learning.

### 2.3. Mã hoá biến phân loại
Trong `main.ipynb` (pipeline chính): `LabelEncoder` cho `code`, `sub_code`, `sub_category`, `horizon` sau khi fit trên **hợp** train+test để tránh unseen category.

---

## 3. Kỹ thuật chọn đặc trưng & feature engineering
### 3.1. Chọn “stable features” theo phân đoạn thời gian
Trong `Data preprocessing.ipynb`, dữ liệu được sắp theo `ts_index` và chia thành 5 phần. Với mỗi phần, một mô hình LightGBM được fit (train/valid tách theo quantile 0.8 của `ts_index`) để lấy feature importance (gain). Sau đó lấy hợp các top features và giữ các đặc trưng xuất hiện ổn định ở **ít nhất 3/5 phần**.

Trong `main.ipynb`, danh sách `stable_num_features` được khai báo thủ công (tương ứng với các stable features đã chọn trước đó).

### 3.2. AutoEncoder để trích xuất “deep features”
`main.ipynb` huấn luyện một AutoEncoder nông (MLP) trên các `stable_num_features`:

- Encoder: $\mathbb{R}^d \rightarrow 16 \rightarrow 8$ (ReLU)
- Decoder: $8 \rightarrow 16 \rightarrow d$ (ReLU)

Sau khi train, vector mã hoá (8 chiều) được dùng làm đặc trưng mới `deep_feat_0..7` và được nối vào tập đặc trưng số.

---

## 4. Thiết kế tập train/validation theo thời gian
Để tránh “data leakage” trong bài toán chuỗi thời gian, `main.ipynb` tách theo mốc thời gian:

- `split_time = max(ts_index) - 100`
- Train: `ts_index <= split_time`
- Valid: lấy một cửa sổ bao phủ đủ chiều dài chuỗi để tạo sequence: `ts_index >= split_time - max_seq_len + 1`

Với `max_seq_len = 60`, mô hình học dựa trên 60 bước thời gian gần nhất cho mỗi `code`.

---

## 5. Mô hình hoá
### 5.1. Dataset dạng sequence
Lớp `HedgeFundDataset` (PyTorch) tạo mẫu đầu vào dạng chuỗi độ dài `seq_len`. Một index hợp lệ là index $t$ sao cho toàn bộ $[t-seq\_len+1,\dots,t]$ thuộc cùng một `code`.

Mỗi sample trả về:

- `x_feat`: tensor đặc trưng số kích thước `(seq_len, num_features)`
- `x_code, x_sub_code, x_sub_cat, x_horizon`: tensor chỉ số phân loại theo thời gian `(seq_len,)`
- `y`: mục tiêu tại thời điểm cuối cửa sổ
- `w`: trọng số tại thời điểm cuối cửa sổ

### 5.2. Mô hình Transformer (wide & deep)
Mô hình `Transformer` trong `main.ipynb`:

- Feature projection: `Linear(num_features → embed_dim)`
- Embedding cho `code`, `sub_code`, `sub_category`, `horizon` (cùng `embed_dim`)
- Ghép 5 embedding theo trục feature: `concat` → `Linear(5*embed_dim → embed_dim)` → GELU → LayerNorm → Dropout
- Positional encoding (sin/cos)
- `TransformerEncoder` (2 layers, `nhead=4`, `norm_first=True`)
- Dự đoán từ bước cuối: `decoder(embed_dim → 1)`
- Nhánh “wide”: `skip_linear(num_features → 1)` dùng `x_features` ở bước cuối

Kết quả cuối:
$$
\hat{y} = \hat{y}_{deep} + \hat{y}_{wide}
$$

### 5.3. Baseline: LightGBM
Phần cuối `main.ipynb` huấn luyện LightGBM (GBDT) với:

- Feature: `ts_index` + `stable_num_features` + các biến phân loại đã encode.
- Early stopping.

---

## 6. Hàm mất mát (loss) & tối ưu
### 6.1. Loss kết hợp Weighted-Normalized MSE và Pearson
Trong `main.ipynb`, `Loss` được định nghĩa:

**(i) Weighted normalized MSE**
$$
L_{mse} = \frac{\sum_i w_i\,(y_i-\hat{y}_i)^2}{\sum_i w_i\,y_i^2 + \epsilon}
$$

**(ii) Pearson correlation loss**
Gọi $\rho$ là hệ số tương quan Pearson giữa $\hat{y}$ và $y$ trong batch:
$$
L_{pearson} = 1 - \rho
$$

**(iii) Loss tổng**
$$
L = \alpha\,L_{mse} + (1-\alpha)\,L_{pearson}
$$
với `alpha = 0.6` và `eps = 1e-8`.

Ngoài ra có một biến thể `Loss2` dùng trọng số chuẩn hoá $w/\sum w$ và tính Pearson theo trọng số.

### 6.2. Tối ưu hoá
Thiết lập trong `main.ipynb`:

- Optimizer: SGD, `lr=1e-3`, `momentum=0.9`, `weight_decay=1e-2`
- Scheduler: `ReduceLROnPlateau(factor=0.3, patience=0)`
- Gradient clipping: `max_norm=1.0`
- Số epoch: 4 

---

## 7. Suy luận & tạo submission
Do test không có đủ lịch sử cho mỗi `code`, notebook nối:

1) `last_days`: lấy `max_seq_len-1` dòng cuối theo mỗi `code` từ train.
2) Ghép với `test_df`, sắp theo `code, ts_index`.
3) Forward-fill đặc trưng theo nhóm `code` và điền phần còn thiếu bằng 0.
4) Tạo `TestHedgeFundDataset` chỉ lấy các cửa sổ kết thúc tại dòng thuộc test.
5) Dự đoán và ghi ra `submission.csv`.

---

## 8. Cấu trúc thư mục
- `Data preprocessing.ipynb`: EDA + cleaning + chọn stable features bằng LightGBM importance theo phân đoạn thời gian.
- `main.ipynb`: pipeline chính (preprocess → AE deep features → Transformer training → inference), kèm baseline LightGBM.
- `README.md`: báo cáo tóm tắt dự án.

---

## 9. Hướng dẫn chạy lại 
### 9.1. Môi trường
Các thư viện xuất hiện trong notebooks:

- `numpy`, `pandas`
- `pyarrow` (đọc parquet)
- `scikit-learn` (`StandardScaler`, `LabelEncoder`)
- `torch` (PyTorch)
- `lightgbm`
- `matplotlib`, `seaborn`

### 9.2. Dữ liệu

Trong Hedge fund - Time series forecasting competition trên kaggle 

### 9.3. Thứ tự chạy
1) Mở `Data preprocessing.ipynb` nếu bạn muốn tái tạo bước chọn stable features.
2) Mở `main.ipynb` và chạy lần lượt các cell để:
	- Load dữ liệu
	- Preprocess + encode
	- Train AutoEncoder (tạo `deep_feat_*`)
	- Train Transformer
	- Generate `submission.csv`

---

## 10. Hạn chế & hướng phát triển

### 10.1. Hạn chế hiện tại: “nút thắt” của Transformer với dữ liệu tài chính
Mặc dù pipeline trong `main.ipynb` sử dụng kiến trúc **Wide & Deep Transformer** (kết hợp embedding ngữ cảnh + self-attention + nhánh linear “wide”), kết quả thực nghiệm **kém hơn** baseline **LightGBM**. Đây không hẳn là do mô hình “yếu”, mà chủ yếu đến từ bản chất của dữ liệu hedge fund/chứng khoán:

- **Tỷ lệ Tín hiệu/Nhiễu rất thấp (Low Signal-to-Noise Ratio — SNR)**
	Dữ liệu tài chính chứa nhiều biến động ngẫu nhiên và nhiễu cao tần. Self-Attention rất mạnh trong việc tìm “pattern”, nhưng ở môi trường SNR thấp, mô hình dễ **học vẹt** các vi biến động ngẫu nhiên (overfit) thay vì nắm bắt quy luật có tính khái quát.

- **Thiếu hụt Data Engineering đặc thù cho dữ liệu bảng theo thời gian**
	Do phần lớn thời gian của dự án dành cho việc cải thiện mô hình transformer, hiện tại đang “đẩy” gánh nặng khai phá quy luật cho AutoEncoder + Attention trên các `feature_*` tương đối thô. 

### 10.2. Hướng phát triển: giảm độ phức tạp mô hình, tập trung vào Data Engineering
Để khắc phục các hạn chế trên, định hướng tiếp theo là **giảm độ “phức tạp mô hình”** và **đầu tư mạnh vào dữ liệu/đặc trưng**, nhằm nâng SNR và tăng tính ổn định theo thời gian.
