# 📊 DỰ BÁO KHÁCH HÀNG RỜI BỎ (TELECOM CUSTOMER CHURN PREDICTION)

## 📁 Thông tin dự án
- **Môn học:** Phân tích dữ liệu kinh doanh (IS403.Q11)[cite: 2].
- **Giảng viên hướng dẫn:** ThS. Dương Phi Long.
- **Đơn vị:** Trường Đại học Công nghệ Thông tin - ĐHQG-HCM (UIT)[cite: 2].
- **Nhóm thực hiện:** Trần Linh Chi - 22520154 (Nhóm trưởng), Phùng Khánh Hoàng - 22520476, Nguyễn Hiếu Nghĩa - 22520948, Huỳnh Quốc Thiên Ân - 22520015, Đặng Việt Hoàng - 22520458.

---

## 🚀 Công nghệ sử dụng
- Ngôn ngữ: Python
- Thư viện: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost.
- Công cụ: Google Colab, GitHub.

---

## 📂 Cấu trúc Repository
- notebooks/: Chứa file code Google Colab xử lý chi tiết từ EDA đến Modeling.
- data/: Thông tin mô tả về tập dữ liệu (Telco Customer Churn).
- images/: Các biểu đồ trực quan hóa, ma trận nhầm lẫn (Confusion Matrix) và Feature Importance.
- docs/: Tài liệu nghiệp vụ và báo cáo kết quả.

---

## 🎯 Bối cảnh & Bài toán kinh doanh
Trong ngành viễn thông, chi phí để thu hút một khách hàng mới thường cao gấp **6 đến 7 lần** so với việc duy trì khách hàng hiện hữu. Với tỷ lệ rời bỏ (Churn) hàng năm lên tới **20-40%** tại các tập đoàn lớn, việc chuyển dịch sang chiến lược "tăng trưởng dựa trên trải nghiệm" là yêu cầu sống còn.

**Mục tiêu cốt lõi:**
- Xây dựng mô hình phân lớp (Classification) dự báo chính xác trạng thái khách hàng.
- Tối ưu hóa chỉ số **Recall** để giảm thiểu sai lầm loại II (False Negative) — tránh bỏ sót những khách hàng sắp rời đi.
- Giải mã "hộp đen" thuật toán bằng **SHAP** để tìm ra nguyên nhân gốc rễ thúc đẩy khách hàng ngưng dịch vụ.

---

## 🛠 Phương pháp thực hiện

### 1. Dữ liệu (Dataset)
Sử dụng bộ dữ liệu **"Customer Churn Dataset"** (Kaggle 2023) bao gồm thông tin nhân khẩu học, hành vi tiêu dùng và thông tin hợp đồng.
  - **Phạm vi dữ liệu:** Thông tin nhân khẩu học, các loại hình dịch vụ đang sử dụng và chi tiết tài khoản hợp đồng.
  - **Chi tiết đặc tả:** Xem thêm tại [data/README.md](https://www.google.com/search?q=./data/README.md).

### 2. Khám phá & Phân tích dữ liệu (EDA)
Đây là giai đoạn quan trọng để thấu hiểu hành vi khách hàng và tìm ra các biến số có tác động mạnh nhất đến quyết định rời bỏ:
- Phân tích đơn biến (Univariate Analysis): Khảo sát sự phân bổ của các biến mục tiêu và nhận diện tình trạng mất cân bằng dữ liệu (Imbalanced Data) – yếu tố then chốt quyết định chiến lược huấn luyện mô hình.
- Phân tích nhị biến (Bivariate Analysis):
  Đối soát Churn với các biến định tính như Loại hợp đồng (Contract), Phương thức thanh toán (Payment Method) để tìm ra các phân khúc khách hàng rủi ro.
  Sử dụng biểu đồ mật độ (Density Plot) để xác định ngưỡng cước phí hàng tháng (Monthly Charges) mà tại đó tỷ lệ rời bỏ bắt đầu tăng mạnh.
- Phân tích đa biến (Multivariate Analysis): Xây dựng ma trận tương quan (Correlation Heatmap) để đánh giá mối quan hệ giữa các biến số như Tenure, Total Spend, và Monthly Charges. Nhận diện và xử lý hiện tượng đa cộng tuyến để đảm bảo tính ổn định cho các mô hình phân loại.

### 3. Tiền xử lý (Preprocessing)
- **Làm sạch:** Loại bỏ nhiễu, xử lý giá trị thiếu và đồng bộ kiểu dữ liệu.
- **Encoding:** Sử dụng **One-hot Encoding** kết hợp kỹ thuật **Drop-first** để tránh bẫy biến giả (Dummy Variable Trap).
- **Chuẩn hóa:** Sử dụng **StandardScaler** để đưa các biến số về cùng một thang đo, đảm bảo sự công bằng và tốc độ hội tụ cho các thuật toán.

### 4. Mô hình hóa & Tối ưu hóa
Thử nghiệm 5 thuật toán: Logistic Regression, KNN, Linear SVC, Random Forest và XGBoost.
- **Tối ưu tham số:** Sử dụng **GridSearchCV** với tiêu chuẩn ưu tiên là **Recall**, nhằm tối đa hóa khả năng nhận diện chính xác khách hàng có rủi ro rời bỏ dịch vụ.
- **Gán trọng số:** Sử dụng `class_weight` và `scale_pos_weight` để xử lý vấn đề mất cân bằng dữ liệu, ép mô hình "nhạy cảm" hơn với nhãn Churn.

---

## 📈 Kết quả thực nghiệm

Mô hình **XGBoost** đạt hiệu suất xuất sắc nhất trong việc bảo toàn doanh thu:
- **Recall (1):** **99.87%** (Chỉ bỏ sót 39 trường hợp trên tổng số hơn 30,000 khách hàng rời mạng thực tế).
- **F1-Score:** 0.655.
- **Chiến lược:** Chấp nhận mức **Accuracy (~50%)** và **Precision (~48%)** trung bình để đổi lấy khả năng nhận diện gần như tuyệt đối khách hàng rủi ro. Điều này phù hợp với thực tế kinh doanh viễn thông: chi phí tặng voucher giữ chân rẻ hơn nhiều so với thiệt hại mất đi một thuê bao.

---

## 💡 Insights từ SHAP (Explainable AI)
Sử dụng kỹ thuật SHAP để minh bạch hóa các quyết định của mô hình:
- **Yếu tố thúc đẩy rời bỏ:** Loại hợp đồng theo tháng (**Monthly Contract**) và Số cuộc gọi hỗ trợ (**Support Calls**) là hai tác nhân mạnh nhất.
- **Yếu tố kìm giữ:** Tổng chi tiêu tích lũy (**Total Spend**) cao tỷ lệ thuận với lòng trung thành của khách hàng.

---

## 🚀 Đề xuất chiến lược (Actionable Insights)
Dựa trên dữ liệu, doanh nghiệp cần:
1. **Cải thiện CSKH:** Tập trung nâng cao chất lượng hỗ trợ kỹ thuật để giảm số lượng cuộc gọi phàn nàn.
2. **Chuyển đổi hợp đồng:** Thiết kế các chương trình ưu đãi để khuyến khích khách hàng chuyển từ hợp đồng ngắn hạn sang cam kết dài hạn.
3. **Cá nhân hóa ưu đãi:** Tập trung nguồn lực giữ chân vào nhóm khách hàng có giá trị tích lũy (Total Spend) cao nhưng bắt đầu có dấu hiệu tăng số cuộc gọi hỗ trợ.

---


