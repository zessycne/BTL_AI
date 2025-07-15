import joblib

# Hàm tải mô hình và vectorizer đã lưu

def tai_mo_hinh_va_vectorizer(duong_dan_mo_hinh: str, duong_dan_vectorizer: str):
    mo_hinh = joblib.load(duong_dan_mo_hinh)
    vectorizer = joblib.load(duong_dan_vectorizer)
    return mo_hinh, vectorizer

# Hàm dự đoán tin nhắn mới

def du_doan_tin_nhan(mo_hinh, vectorizer, tin_nhan: str):
    dac_trung = vectorizer.transform([tin_nhan])
    du_doan = mo_hinh.predict(dac_trung)[0]
    return 'Thư rác' if du_doan == 1 else 'Không phải thư rác' 