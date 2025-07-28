import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from tien_xu_ly import doc_va_tien_xu_ly_du_lieu
from dac_trung import trich_xuat_tfidf
import joblib
import time

# Pipeline huấn luyện và đánh giá mô hình Logistic Regression + TF-IDF

def xay_dung_va_danh_gia_mo_hinh(duong_dan_file: str):
    X_train, X_test, y_train, y_test = doc_va_tien_xu_ly_du_lieu(duong_dan_file)
    X_train_tfidf, X_test_tfidf, vectorizer = trich_xuat_tfidf(X_train, X_test)
    mo_hinh = LogisticRegression(max_iter=1000)
    mo_hinh.fit(X_train_tfidf, y_train)
    y_du_doan = mo_hinh.predict(X_test_tfidf)
    do_chinh_xac = accuracy_score(y_test, y_du_doan)
    bao_cao = classification_report(y_test, y_du_doan, target_names=["Không spam", "Spam"])
    print(f"Độ chính xác của mô hình: {do_chinh_xac:.2f}")
    print("\nBáo cáo phân loại:")
    print(bao_cao)
    return mo_hinh, vectorizer

def du_doan_tin_nhan(mo_hinh, vectorizer, tin_nhan: str):
    tin_nhan_tfidf = vectorizer.transform([tin_nhan])
    du_doan = mo_hinh.predict(tin_nhan_tfidf)[0]
    return "Spam" if du_doan == 1 else "Không spam"

def luu_mo_hinh_va_vectorizer(mo_hinh, vectorizer, duong_dan_mo_hinh: str, duong_dan_vectorizer: str):
    joblib.dump(mo_hinh, duong_dan_mo_hinh)
    joblib.dump(vectorizer, duong_dan_vectorizer)

def tai_mo_hinh_va_vectorizer(duong_dan_mo_hinh: str, duong_dan_vectorizer: str):
    mo_hinh = joblib.load(duong_dan_mo_hinh)
    vectorizer = joblib.load(duong_dan_vectorizer)
    return mo_hinh, vectorizer

def huan_luyen_mo_hinh(X_train_tfidf, y_train):
    mo_hinh = LogisticRegression(max_iter=1000)
    mo_hinh.fit(X_train_tfidf, y_train)
    return mo_hinh

def danh_gia_mo_hinh(mo_hinh, X_test_tfidf, y_test):
    du_doan = mo_hinh.predict(X_test_tfidf)
    do_chinh_xac = accuracy_score(y_test, du_doan)
    bao_cao = classification_report(y_test, du_doan, target_names=["Không phải rác", "Thư rác"])
    return do_chinh_xac, bao_cao

def train_and_evaluate(duong_dan_file: str, duong_dan_mo_hinh: str, duong_dan_vectorizer: str):
    start_time = time.time()
    X_train, X_test, y_train, y_test = doc_va_tien_xu_ly_du_lieu(duong_dan_file)
    X_train_tfidf, X_test_tfidf, vectorizer = trich_xuat_tfidf(X_train, X_test)
    mo_hinh = huan_luyen_mo_hinh(X_train_tfidf, y_train)
    do_chinh_xac, bao_cao = danh_gia_mo_hinh(mo_hinh, X_test_tfidf, y_test)
    from sklearn.metrics import precision_score, recall_score, f1_score
    y_pred = mo_hinh.predict(X_test_tfidf)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Độ chính xác: {do_chinh_xac:.4f}')
    print('Báo cáo phân loại:')
    print(bao_cao.strip())
    luu_mo_hinh_va_vectorizer(mo_hinh, vectorizer, duong_dan_mo_hinh, duong_dan_vectorizer)
    end_time = time.time()
    print(f'Đã lưu mô hình vào {duong_dan_mo_hinh} và vectorizer vào {duong_dan_vectorizer}')
    print(f'Thời gian chạy: {end_time - start_time:.2f} giây')

if __name__ == '__main__':
    train_and_evaluate('spam.csv', 'mo_hinh_spam_tfidf.pkl', 'vectorizer_spam.pkl') 