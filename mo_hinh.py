import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from tien_xu_ly import doc_va_tien_xu_ly_du_lieu
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Xây dựng mô hình phân loại tin nhắn spam

def xay_dung_va_danh_gia_mo_hinh(duong_dan_file: str):
    # Đọc và tiền xử lý dữ liệu
    X_train, X_test, y_train, y_test = doc_va_tien_xu_ly_du_lieu(duong_dan_file)

    # Biến đổi văn bản sang vector đặc trưng TF-IDF
    bo_vector_hoa = TfidfVectorizer(ngram_range=(1,2), max_features=3000)
    X_train_vector = bo_vector_hoa.fit_transform(X_train)
    X_test_vector = bo_vector_hoa.transform(X_test)

    # Khởi tạo và huấn luyện mô hình Logistic Regression
    mo_hinh = LogisticRegression(max_iter=1000)
    mo_hinh.fit(X_train_vector, y_train)

    # Dự đoán trên tập kiểm tra
    y_du_doan = mo_hinh.predict(X_test_vector)

    # Đánh giá mô hình
    do_chinh_xac = accuracy_score(y_test, y_du_doan)
    bao_cao = classification_report(y_test, y_du_doan, target_names=["Không spam", "Spam"])

    print(f"Độ chính xác của mô hình: {do_chinh_xac:.2f}")
    print("\nBáo cáo phân loại:")
    print(bao_cao)

    return mo_hinh, bo_vector_hoa

# Hàm dự đoán tin nhắn mới
def du_doan_tin_nhan(mo_hinh, bo_vector_hoa, tin_nhan: str):
    tin_nhan_vector = bo_vector_hoa.transform([tin_nhan])
    du_doan = mo_hinh.predict(tin_nhan_vector)[0]
    return "Spam" if du_doan == 1 else "Không spam"

# Hàm lưu mô hình và vectorizer

def luu_mo_hinh_va_vectorizer(mo_hinh, vectorizer, duong_dan_mo_hinh: str, duong_dan_vectorizer: str):
    joblib.dump(mo_hinh, duong_dan_mo_hinh)
    joblib.dump(vectorizer, duong_dan_vectorizer)

# Hàm tải mô hình và vectorizer
def tai_mo_hinh(duong_dan_mo_hinh: str, duong_dan_vector: str):
    mo_hinh = joblib.load(duong_dan_mo_hinh)
    bo_vector_hoa = joblib.load(duong_dan_vector)
    return mo_hinh, bo_vector_hoa

# Hàm huấn luyện mô hình phân loại thư rác

def huan_luyen_mo_hinh(X_train_tfidf, y_train):
    # Khởi tạo mô hình Naive Bayes đa thức
    mo_hinh = MultinomialNB()
    # Huấn luyện mô hình
    mo_hinh.fit(X_train_tfidf, y_train)
    return mo_hinh

# Hàm đánh giá mô hình

def danh_gia_mo_hinh(mo_hinh, X_test_tfidf, y_test):
    # Dự đoán trên tập kiểm tra
    du_doan = mo_hinh.predict(X_test_tfidf)
    # Tính toán độ chính xác và báo cáo phân loại
    do_chinh_xac = accuracy_score(y_test, du_doan)
    bao_cao = classification_report(y_test, du_doan, target_names=['Không phải rác', 'Thư rác'])
    return do_chinh_xac, bao_cao

def train_and_evaluate(duong_dan_file: str, duong_dan_mo_hinh: str, duong_dan_vectorizer: str):
    """
    Hàm thực hiện toàn bộ quy trình training, test và lưu mô hình, vectorizer.
    """
    from tien_xu_ly import doc_va_tien_xu_ly_du_lieu
    from dac_trung import trich_xuat_tfidf
    # Đọc và tiền xử lý dữ liệu, chia train/test
    X_train, X_test, y_train, y_test = doc_va_tien_xu_ly_du_lieu(duong_dan_file)
    # Trích xuất đặc trưng TF-IDF
    X_train_tfidf, X_test_tfidf, bo_tfidf = trich_xuat_tfidf(X_train, X_test)
    # Huấn luyện mô hình
    mo_hinh = huan_luyen_mo_hinh(X_train_tfidf, y_train)
    # Đánh giá mô hình
    do_chinh_xac, bao_cao = danh_gia_mo_hinh(mo_hinh, X_test_tfidf, y_test)
    print(f'Độ chính xác: {do_chinh_xac:.4f}')
    print('Báo cáo phân loại:')
    print(bao_cao)
    # Lưu mô hình và vectorizer
    luu_mo_hinh_va_vectorizer(mo_hinh, bo_tfidf, duong_dan_mo_hinh, duong_dan_vectorizer)
    print(f'Đã lưu mô hình vào {duong_dan_mo_hinh} và vectorizer vào {duong_dan_vectorizer}')

if __name__ == '__main__':
    # Chạy quy trình training/test chính thống
    train_and_evaluate('spam.csv', 'mo_hinh_spam.pkl', 'vectorizer_spam.pkl')
