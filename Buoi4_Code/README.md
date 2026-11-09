# Demo Streamlit: phân loại khối trên ảnh nhũ ảnh

Chạy từ thư mục dự án:

```bash
source myenv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Mở http://localhost:8501, tải ảnh JPG/PNG vùng khối đã cắt hoặc chọn ảnh mẫu tập test rồi bấm **Dự đoán**.
Ảnh mẫu chỉ khả dụng khi có các CSV và ảnh JPEG tương ứng trong `Data/`.

Ứng dụng đọc `models/svm_pipeline.pkl`. Nếu thiếu pipeline, ứng dụng ghép hai file
`scaler_pickle.pkl` và `svm_model_pickle.pkl`. Chỉ dùng pickle tin cậy do bạn huấn luyện.
Sau khi huấn luyện lại và ghi đè mô hình, khởi động lại Streamlit để nạp mô hình mới.

Tiền xử lý giống `SVM_Breast_Cancer.ipynb`: lấy trung bình RGB thành ảnh xám,
resize 128 × 128, HOG (ô 16 × 16, khối 2 × 2), ghép mean và std thành 1.766 đặc trưng,
rồi đưa qua scaler và SVM. Nhãn 0 là lành tính; nhãn 1 là ác tính.
Mô hình chưa bật `probability`, vì vậy demo không hiển thị phần trăm tin cậy.

Đây là demo học tập, không dùng để chẩn đoán y khoa. Đầu vào cần là vùng khối đã cắt,
giống ảnh train, thay vì ảnh nhũ ảnh toàn cảnh.
