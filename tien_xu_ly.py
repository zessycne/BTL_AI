import pandas as pd
from sklearn.model_selection import train_test_split

# Hàm đọc và tiền xử lý dữ liệu

def doc_va_tien_xu_ly_du_lieu(duong_dan_file: str):
    # Thử đọc với utf-8, nếu lỗi thì dùng latin1
    try:
        du_lieu = pd.read_csv(duong_dan_file, encoding='utf-8')
    except UnicodeDecodeError:
        du_lieu = pd.read_csv(duong_dan_file, encoding='latin1')
    # Đổi tên cột cho dễ xử lý
    du_lieu = du_lieu.rename(columns={'v1': 'nhan', 'v2': 'noi_dung'})
    # Chỉ giữ 2 cột cần thiết
    du_lieu = du_lieu[['nhan', 'noi_dung']]
    # Loại bỏ dòng bị thiếu dữ liệu
    du_lieu = du_lieu.dropna()
    # Đảm bảo cột 'nhan' là Series, dùng .replace đúng chuẩn
    du_lieu['nhan'] = pd.Series(du_lieu['nhan']).astype(str).replace({'ham': 0, 'spam': 1})
    # Tách tập train/test
    X_train, X_test, y_train, y_test = train_test_split(
        du_lieu['noi_dung'], du_lieu['nhan'], test_size=0.2, random_state=42, stratify=du_lieu['nhan']
    )
    return X_train, X_test, y_train, y_test 