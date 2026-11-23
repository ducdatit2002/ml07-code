"""Hàm dùng chung giữa notebook và giao diện Streamlit."""
import re  # [H01] Thư viện biểu thức chính quy, dùng làm sạch văn bản.

LABEL_NAMES = {  # [H02] Đổi mã chủ đề arXiv sang tên dễ đọc.
    "astro-ph": "Astrophysics (Vật lý thiên văn)",
    "cond-mat": "Condensed Matter (Vật chất ngưng tụ)",
    "cs": "Computer Science (Khoa học máy tính)",
    "math": "Mathematics (Toán học)",
    "physics": "Physics (Vật lý)",
}


def main_category(categories):  # [H03] Lấy nhãn arXiv đầu tiên, rút gọn về nhóm chính.
    if not isinstance(categories, str) or not categories.strip():  # [H04] Phòng nhãn rỗng/None.
        return None  # [H05] Không có nhãn hợp lệ.
    first = categories.split()[0]  # [H06] 'math.CO cs.CG' thành 'math.CO'.
    return first.split(".")[0]  # [H07] 'math.CO' thành 'math'.


def clean_text(text):  # [H08] Chuẩn hóa abstract trước khi vector hóa.
    if not isinstance(text, str):  # [H09] Tránh lỗi khi text là None.
        return ""  # [H10] Trả về chuỗi rỗng.
    text = text.strip().replace("\n", " ").replace("\r", " ")  # [H11] Xóa khoảng trắng biên, thay xuống dòng.
    text = re.sub(r"[^\w\s]", " ", text)  # [H12] Thay dấu câu bằng khoảng trắng.
    text = re.sub(r"\d+", " ", text)  # [H13] Bỏ chữ số theo quy trình của bài học.
    text = re.sub(r"_+", " ", text)  # [H14] Bỏ gạch dưới.
    text = re.sub(r"\s+", " ", text)  # [H15] Gom các khoảng trắng liên tiếp thành một.
    return text.lower().strip()  # [H16] Viết thường và trả chuỗi đã làm sạch.
