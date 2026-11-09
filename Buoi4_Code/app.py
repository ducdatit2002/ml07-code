"""Chạy: python -m streamlit run app.py."""

import io
import pickle
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from skimage.feature import hog
from skimage.transform import resize

ROOT = Path(__file__).resolve().parent
LABELS = {0: "Lành tính (Benign)", 1: "Ác tính (Malignant)"}


@st.cache_resource
def load_model():
    """Chỉ đọc các pickle đã huấn luyện trong thư mục models của dự án."""
    folder = ROOT / "models"
    pipeline = folder / "svm_pipeline.pkl"
    if pipeline.is_file():
        with pipeline.open("rb") as handle:
            return pickle.load(handle)
    with (folder / "scaler_pickle.pkl").open("rb") as handle:
        scaler = pickle.load(handle)
    with (folder / "svm_model_pickle.pkl").open("rb") as handle:
        svm = pickle.load(handle)
    return Pipeline([("scaler", scaler), ("svm", svm)])


def extract_features(image):
    """Giữ nguyên cách xử lý và thứ tự đặc trưng trong notebook train."""
    if image.ndim == 3:
        image = np.mean(image[..., :3], axis=2)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("Ảnh phải là ảnh xám hoặc ảnh RGB hợp lệ.")
    resized = resize(image, (128, 128), anti_aliasing=True)
    descriptor = hog(
        resized, pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), feature_vector=True,
    )
    features = np.concatenate([descriptor, [resized.mean(), resized.std()]])
    if not np.isfinite(features).all():
        raise ValueError("Ảnh chứa giá trị pixel không hợp lệ.")
    return features.reshape(1, -1), resized


@st.cache_data
def test_samples():
    """Lấy tối đa 20 ảnh crop có trên máy từ tập test gốc."""
    data = ROOT / "Data"
    cases = data / "mass_case_description_test_set.csv"
    metadata = data / "dicom_info.csv"
    if not cases.is_file() or not metadata.is_file():
        return []
    dicom = pd.read_csv(metadata, usecols=["SeriesDescription", "image_path"])
    lookup = {}
    for path in dicom.loc[dicom.SeriesDescription == "cropped images", "image_path"]:
        parts = str(path).strip().split("/")
        if len(parts) >= 2:
            lookup[parts[-2]] = data / "jpeg" / parts[-2] / parts[-1]
    samples = []
    for _, row in pd.read_csv(cases).iterrows():
        parts = str(row["cropped image file path"]).strip().split("/")
        path = lookup.get(parts[-2]) if len(parts) >= 2 else None
        if path is not None and path.is_file():
            samples.append({
                "path": str(path), "pathology": row["pathology"],
                "title": f"{row['patient_id']} · {row['left or right breast']} · "
                         f"{row['image view']} · khối {row['abnormality id']}",
            })
        if len(samples) == 20:
            break
    return samples


def main():
    st.set_page_config(page_title="Demo SVM ảnh nhũ ảnh", page_icon="🔬", layout="wide")
    st.title("Phân loại khối trên ảnh nhũ ảnh")
    st.write("Demo mô hình SVM đã huấn luyện: phân loại lành tính và ác tính.")
    st.info("Dùng ảnh vùng khối đã cắt (cropped mammogram), giống dữ liệu huấn luyện. "
            "Ứng dụng phục vụ học tập, không dùng để chẩn đoán y khoa.")
    try:
        model = load_model()
    except Exception as exc:
        st.error(f"Không thể nạp mô hình trong models/: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Mô hình đang sử dụng")
        st.write("StandardScaler → SVM (RBF)")
        st.caption("Ảnh xám → 128 × 128 → HOG + mean + std")
        st.write(f"Số đặc trưng: {model.n_features_in_:,}")
        st.caption("Nhãn 0: lành tính · Nhãn 1: ác tính")

    source = st.radio("Nguồn ảnh", ["Tải ảnh lên", "Ảnh mẫu tập test"], horizontal=True)
    sample = None
    image = None
    try:
        if source == "Tải ảnh lên":
            uploaded = st.file_uploader("Chọn ảnh vùng khối (JPG / PNG)", type=["jpg", "jpeg", "png"])
            if uploaded is not None:
                # Matplotlib đọc PNG dạng float, JPEG dạng uint8 như lúc train.
                fmt = "png" if uploaded.name.lower().endswith(".png") else "jpeg"
                image = mpimg.imread(io.BytesIO(uploaded.getvalue()), format=fmt)
        else:
            samples = test_samples()
            if not samples:
                st.warning("Không tìm thấy ảnh test trên máy. Hãy dùng chức năng tải ảnh lên.")
                return
            index = st.selectbox("Chọn ảnh mẫu", range(len(samples)),
                                 format_func=lambda value: samples[value]["title"])
            sample = samples[index]
            image = mpimg.imread(sample["path"])
    except Exception as exc:
        st.error(f"Không đọc được ảnh hoặc dữ liệu mẫu: {exc}")
        return

    if image is None:
        st.caption("Chọn ảnh để xem trước và chạy dự đoán.")
        return

    left, right = st.columns(2)
    with left:
        st.subheader("Ảnh đầu vào")
        st.image(image, clamp=True, caption=f"Kích thước: {image.shape[1]} × {image.shape[0]} pixel")
    with right:
        if st.button("Dự đoán", type="primary"):
            try:
                with st.spinner("Đang trích xuất đặc trưng và dự đoán…"):
                    features, processed = extract_features(image)
                    if features.shape[1] != model.n_features_in_:
                        raise ValueError("Số đặc trưng không khớp với mô hình đã lưu.")
                    prediction = int(model.predict(features)[0])
                st.subheader("Kết quả dự đoán")
                st.metric("Nhãn mô hình", LABELS[prediction])
                if sample is not None:
                    st.write(f"Nhãn thật trong tập test: **{sample['pathology']}**")
                    actual = 1 if sample["pathology"] == "MALIGNANT" else 0
                    st.write("Dự đoán khớp nhãn thật." if prediction == actual
                             else "Dự đoán khác nhãn thật.")
                st.caption("Mô hình hiện tại chưa bật probability nên không hiển thị xác suất dự đoán.")
                with st.expander("Xem ảnh sau tiền xử lý"):
                    st.image(processed, clamp=True, caption="Ảnh xám 128 × 128")
            except Exception as exc:
                st.error(f"Không thể dự đoán: {exc}")


if __name__ == "__main__":
    main()
