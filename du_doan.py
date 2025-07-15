import joblib
from mo_hinh import tai_mo_hinh, clean_text_list, batch_encode

# Hàm tải mô hình và vectorizer đã lưu

def tai_mo_hinh_va_vectorizer(duong_dan_mo_hinh: str, duong_dan_vectorizer: str):
    mo_hinh = joblib.load(duong_dan_mo_hinh)
    vectorizer = joblib.load(duong_dan_vectorizer)
    return mo_hinh, vectorizer

# Hàm dự đoán tin nhắn mới

def du_doan_tin_nhan(mo_hinh, embedder, tin_nhan: str):
    """Dự đoán một tin nhắn là spam hay không spam."""
    tin_nhan_clean = clean_text_list([tin_nhan])
    tin_nhan_emb = batch_encode(embedder, tin_nhan_clean)
    du_doan = mo_hinh.predict(tin_nhan_emb)[0]
    return "Spam" if du_doan == 1 else "Không spam"

if __name__ == "__main__":
    mo_hinh, embedder = tai_mo_hinh('mo_hinh_spam.pkl', 'sentence_model.txt')
    tin_nhan = input("Nhập nội dung tin nhắn: ")
    ket_qua = du_doan_tin_nhan(mo_hinh, embedder, tin_nhan)
    print("Kết quả dự đoán:", ket_qua) 