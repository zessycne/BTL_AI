import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from tien_xu_ly import doc_va_tien_xu_ly_du_lieu
import joblib
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

def encode_sentences(model, sentences):
    """Chuyển list câu thành embedding numpy array bằng SentenceTransformer."""
    return np.array(model.encode(sentences, show_progress_bar=False))

def clean_text_list(series):
    """Làm sạch dữ liệu đầu vào: loại bỏ None/NaN, chuyển thành chuỗi, thay thế chuỗi rỗng."""
    return [str(s) if pd.notnull(s) and str(s).strip() != "" else "[EMPTY]" for s in series]

def batch_encode(model, texts, batch_size=128):
    """Encode embedding theo batch nhỏ để tránh tràn bộ nhớ."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False)
        embeddings.append(emb)
    return np.vstack(embeddings)

def xay_dung_va_danh_gia_mo_hinh(duong_dan_file: str):
    """Huấn luyện và đánh giá mô hình phân loại thư rác với SentenceTransformer."""
    X_train, X_test, y_train, y_test = doc_va_tien_xu_ly_du_lieu(duong_dan_file)
    embedder = SentenceTransformer(MODEL_NAME)
    X_train_clean = clean_text_list(X_train)
    X_test_clean = clean_text_list(X_test)
    X_train_emb = batch_encode(embedder, X_train_clean)
    X_test_emb = batch_encode(embedder, X_test_clean)
    mo_hinh = LogisticRegression(max_iter=1000)
    mo_hinh.fit(X_train_emb, y_train)
    y_du_doan = mo_hinh.predict(X_test_emb)
    do_chinh_xac = accuracy_score(y_test, y_du_doan)
    bao_cao = classification_report(y_test, y_du_doan, target_names=["Không spam", "Spam"])
    print(f"Độ chính xác của mô hình: {do_chinh_xac:.2f}")
    print("\nBáo cáo phân loại:")
    print(bao_cao)
    return mo_hinh, embedder

def du_doan_tin_nhan(mo_hinh, embedder, tin_nhan: str):
    """Dự đoán một tin nhắn/email là spam hay không spam."""
    tin_nhan_clean = clean_text_list([tin_nhan])
    tin_nhan_emb = batch_encode(embedder, tin_nhan_clean)
    du_doan = mo_hinh.predict(tin_nhan_emb)[0]
    return "Spam" if du_doan == 1 else "Không spam"

def luu_mo_hinh_va_embedder(mo_hinh, duong_dan_mo_hinh: str, duong_dan_embedder: str):
    """Lưu mô hình và tên model embedding vào file."""
    joblib.dump(mo_hinh, duong_dan_mo_hinh)
    with open(duong_dan_embedder, 'w', encoding='utf-8') as f:
        f.write(MODEL_NAME)

def tai_mo_hinh(duong_dan_mo_hinh: str, duong_dan_embedder: str):
    """Tải mô hình và SentenceTransformer từ file."""
    mo_hinh = joblib.load(duong_dan_mo_hinh)
    with open(duong_dan_embedder, 'r', encoding='utf-8') as f:
        model_name = f.read().strip()
    embedder = SentenceTransformer(model_name)
    return mo_hinh, embedder

def huan_luyen_mo_hinh(X_train_emb, y_train):
    """Huấn luyện mô hình Logistic Regression với embedding."""
    mo_hinh = LogisticRegression(max_iter=1000)
    mo_hinh.fit(X_train_emb, y_train)
    return mo_hinh

def danh_gia_mo_hinh(mo_hinh, X_test_emb, y_test):
    """Đánh giá mô hình trên tập test."""
    du_doan = mo_hinh.predict(X_test_emb)
    do_chinh_xac = accuracy_score(y_test, du_doan)
    bao_cao = classification_report(y_test, du_doan, target_names=['Không phải rác', 'Thư rác'])
    return do_chinh_xac, bao_cao

def train_and_evaluate(duong_dan_file: str, duong_dan_mo_hinh: str, duong_dan_embedder: str):
    """Pipeline: train, test, lưu mô hình và tên model embedding."""
    X_train, X_test, y_train, y_test = doc_va_tien_xu_ly_du_lieu(duong_dan_file)
    embedder = SentenceTransformer(MODEL_NAME)
    X_train_clean = clean_text_list(X_train)
    X_test_clean = clean_text_list(X_test)
    X_train_emb = batch_encode(embedder, X_train_clean)
    X_test_emb = batch_encode(embedder, X_test_clean)
    mo_hinh = huan_luyen_mo_hinh(X_train_emb, y_train)
    do_chinh_xac, bao_cao = danh_gia_mo_hinh(mo_hinh, X_test_emb, y_test)
    print(f'Độ chính xác: {do_chinh_xac:.4f}')
    print('Báo cáo phân loại:')
    print(bao_cao)
    luu_mo_hinh_va_embedder(mo_hinh, duong_dan_mo_hinh, duong_dan_embedder)
    print(f'Đã lưu mô hình vào {duong_dan_mo_hinh} và tên model SentenceTransformer vào {duong_dan_embedder}')

if __name__ == '__main__':
    train_and_evaluate('spam.csv', 'mo_hinh_spam.pkl', 'sentence_model.txt')
