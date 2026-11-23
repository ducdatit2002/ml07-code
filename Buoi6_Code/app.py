"""Giao diện phân loại abstract arXiv bằng KNN. Chạy: python -m streamlit run app.py"""
from pathlib import Path  # [S01] Làm việc với thư mục và đường dẫn file.
import joblib  # [S02] Đọc model và vectorizer đã được lưu từ Colab.
import pandas as pd  # [S03] Chuẩn bị bảng dữ liệu cho biểu đồ.
import streamlit as st  # [S04] Xây dựng giao diện web bằng Python.
from helpers import LABEL_NAMES, clean_text  # [S05] Dùng chung xử lý chữ và tên nhãn.

BASE_DIR = Path(__file__).resolve().parent  # [S06] Xác định thư mục chứa app.py.
MODEL_DIR = BASE_DIR / "models"  # [S07] Mô hình được đặt trong thư mục models.
FILE_NAMES = {"TF-IDF": "tfidf_knn.joblib", "Bag of Words": "bow_knn.joblib"}  # [S08] Ánh xạ lựa chọn sang tệp mô hình.

st.set_page_config(page_title="Phân loại bài báo khoa học", page_icon="📚", layout="wide")  # [S09] Cấu hình trang.
st.title("📚 Phân loại bài báo khoa học với KNN")  # [S10] Tiêu đề chính.
st.caption("Bài thực hành Buổi 6: NLP, Bag of Words, TF-IDF và K-Nearest Neighbors")  # [S11] Mô tả ngắn.

with st.sidebar:  # [S12] Mở khu vực cấu hình bên trái.
    st.header("⚙️ Cấu hình")  # [S13] Tiêu đề sidebar.
    method = st.selectbox("Phương pháp vector hóa", list(FILE_NAMES))  # [S14] Chọn TF-IDF hoặc Bag of Words.
    st.selectbox("Mô hình", ["K-Nearest Neighbors (KNN)"], disabled=True)  # [S15] Mô hình duy nhất của buổi học.
    show_scores = st.checkbox("Hiển thị biểu đồ điểm", value=True)  # [S16] Bật/tắt biểu đồ.
    show_terms = st.checkbox("Hiển thị từ khóa", value=True)  # [S17] Bật/tắt bảng đặc trưng.

model_path = MODEL_DIR / FILE_NAMES[method]  # [S18] Tạo đường dẫn đến mô hình theo lựa chọn.
if not model_path.exists():  # [S19] Kiểm tra model đã được xuất từ notebook chưa.
    st.error(f"Chưa tìm thấy {model_path.name} trong thư mục models/.")  # [S20] Báo lỗi thiếu file.
    st.info("Hãy chạy notebook để tạo file .joblib và đặt vào thư mục models/.")  # [S21] Hướng dẫn khắc phục.
    st.stop()  # [S22] Dừng chương trình, tránh dự đoán khi không có model.

@st.cache_resource  # [S23] Cache model để không đọc lại khi người dùng thao tác.
def load_bundle(path):  # [S24] Hàm mở tệp chứa cả model, vectorizer và nhãn.
    return joblib.load(path)  # [S25] Chỉ load file do chính bạn huấn luyện và tin cậy.

bundle = load_bundle(str(model_path))  # [S26] Nạp bộ mô hình đang chọn.
model = bundle["model"]  # [S27] Lấy thuật toán KNN đã fit.
vectorizer = bundle["vectorizer"]  # [S28] Lấy bộ từ điển đã fit ở tập train.
id_to_label = bundle["id_to_label"]  # [S29] Bảng ánh xạ từ ID số sang mã nhãn.
meta = bundle.get("meta", {})  # [S30] Thông tin đánh giá và nguồn dữ liệu (nếu có).

st.success(f"Đã nạp {method} + KNN thành công")  # [S31] Xác nhận quá trình nạp hoàn tất.
col1, col2, col3 = st.columns(3)  # [S32] Tạo ba ô thông tin nằm cạnh nhau.
col1.metric("Mô hình", "KNN")  # [S33] Ô thứ nhất hiển thị thuật toán.
col2.metric("Số chủ đề", len(id_to_label))  # [S34] Ô thứ hai hiển thị số lớp.
col3.metric("Test Accuracy", f"{meta['accuracy']:.1%}" if "accuracy" in meta else "Chưa có")  # [S35] Ô thứ ba hiển thị độ chính xác đo trên test.

sample = (  # [S36] Một abstract ví dụ để người học thử ngay.
    "We propose a neural network for image classification and natural language "
    "processing. Our learning algorithm improves prediction accuracy on datasets."
)
text_input = st.text_area("Nhập abstract tiếng Anh", value=sample, height=180)  # [S37] Ô nhập văn bản nhiều dòng.
clicked = st.button("🔎 Phân loại", type="primary")  # [S38] Nút bắt đầu dự đoán.

if clicked:  # [S39] Chỉ phân loại khi người dùng bấm nút.
    cleaned = clean_text(text_input)  # [S40] Áp dụng cùng quy trình làm sạch như notebook.
    if len(cleaned.split()) < 5:  # [S41] Không nhận văn bản quá ngắn.
        st.warning("Hãy nhập ít nhất 5 từ có nghĩa bằng tiếng Anh.")  # [S42] Yêu cầu nhập lại.
        st.stop()  # [S43] Không chạy tiếp nếu không đạt yêu cầu.
    with st.spinner("Đang vector hóa và tìm các hàng xóm gần nhất..."):  # [S44] Hiệu ứng chờ.
        vector = vectorizer.transform([cleaned])  # [S45] Chỉ transform; tuyệt đối KHÔNG fit lại vectorizer.
        if vector.nnz == 0:  # [S46] Kiểm tra có từ nào nằm trong từ vựng đã học không.
            st.warning("Bài báo không chứa từ vựng mà mô hình đã học.")  # [S47] Báo đầu vào ngoài bộ từ vựng.
            st.stop()  # [S48] Dừng dự đoán.
        predicted_id = int(model.predict(vector)[0])  # [S49] KNN trả về ID của nhóm được dự đoán.
        scores = model.predict_proba(vector)[0]  # [S50] Lấy mức ủng hộ theo phiếu bầu hàng xóm.

    code = id_to_label[predicted_id]  # [S51] Chuyển ID dự đoán sang mã nhóm.
    st.subheader("Kết quả")  # [S52] Tiêu đề phần kết quả.
    st.markdown(f"### {LABEL_NAMES.get(code, code)}")  # [S53] Hiển thị tên nhóm thân thiện.
    score_rows = [  # [S54] Ghép từng lớp với điểm của lớp ấy.
        {"Chủ đề": LABEL_NAMES.get(id_to_label[int(cls)], str(cls)), "Điểm (%)": round(float(p) * 100, 2)}
        for cls, p in zip(model.classes_, scores)  # [S55] classes_ và scores có cùng thứ tự.
    ]
    score_rows.sort(key=lambda row: row["Điểm (%)"], reverse=True)  # [S56] Sắp xếp điểm từ cao xuống thấp.
    st.metric("Điểm ủng hộ cao nhất", f"{score_rows[0]['Điểm (%)']:.1f}%")  # [S57] Hiển thị điểm cao nhất.
    st.caption("Điểm này phản ánh phiếu bầu có trọng số của KNN, không phải xác suất đúng đã hiệu chuẩn.")  # [S58] Giải thích đúng ý nghĩa điểm.

    if show_scores:  # [S59] Chỉ vẽ khi người dùng bật checkbox.
        chart_data = pd.DataFrame(score_rows).set_index("Chủ đề")  # [S60] Đưa dữ liệu về dạng bảng và chọn nhãn hàng.
        st.bar_chart(chart_data)  # [S61] Vẽ biểu đồ so sánh mức ủng hộ của 5 nhóm.

    if show_terms:  # [S62] Chỉ hiển thị bảng từ khóa khi người dùng muốn.
        feature_names = vectorizer.get_feature_names_out()  # [S63] Lấy từ điển đặc trưng đã học.
        sparse_row = vector.tocoo()  # [S64] Lấy các đặc trưng khác 0, tránh chuyển ma trận sang dense.
        terms = sorted(  # [S65] Sắp xếp theo trọng số có trong bài nhập.
            zip((feature_names[j] for j in sparse_row.col), sparse_row.data),
            key=lambda item: item[1], reverse=True
        )[:10]  # [S66] Giữ 10 từ/cụm từ có trọng số lớn nhất.
        st.subheader("Từ / cụm từ nổi bật")  # [S67] Tiêu đề bảng đặc trưng.
        st.dataframe(pd.DataFrame(terms, columns=["Từ / cụm từ", "Trọng số"]), hide_index=True)  # [S68] Vẽ bảng.
        st.caption("Đây là trọng số đầu vào, không phải giải thích nhân quả cho quyết định của KNN.")  # [S69] Tránh diễn giải sai.

    with st.expander("Xem văn bản sau tiền xử lý"):  # [S70] Mở rộng nếu người học muốn kiểm tra.
        st.write(cleaned)  # [S71] In kết quả làm sạch để đối chiếu.
