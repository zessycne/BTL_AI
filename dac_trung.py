from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from typing import Tuple

# Hàm trích xuất đặc trưng TF-IDF

def trich_xuat_tfidf(X_train, X_test) -> Tuple[csr_matrix, csr_matrix, TfidfVectorizer]:
    # Đảm bảo đầu vào là list hoặc Series
    X_train = list(X_train)
    X_test = list(X_test)
    # Khởi tạo bộ biến đổi TF-IDF
    bo_tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=3000)
    # Học và biến đổi dữ liệu train
    X_train_tfidf = bo_tfidf.fit_transform(X_train)
    # Biến đổi dữ liệu test
    X_test_tfidf = bo_tfidf.transform(X_test)
    # Ép kiểu trả về để linter không báo lỗi
    return csr_matrix(X_train_tfidf), csr_matrix(X_test_tfidf), bo_tfidf 