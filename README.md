<<<<<<< HEAD
# 📧 Hệ Thống Nhận Diện Email Spam

Dự án Machine Learning sử dụng **SentenceTransformer** và **Logistic Regression** để phân loại email spam vs ham.

## 🎯 Mục tiêu

Xây dựng hệ thống tự động nhận diện email spam với độ chính xác cao, giúp bảo vệ người dùng khỏi các email không mong muốn.

## 📋 Cấu trúc dự án

```
DemoAI/
├── spam_detection_complete.py    # File Python chính (có thể chạy trực tiếp)
├── demo_spam_detection.txt       # File text chứa code và hướng dẫn
├── spam.csv                      # Dữ liệu huấn luyện
├── mo_hinh_spam.pkl             # Mô hình đã huấn luyện (sẽ tạo)
├── sentence_model.txt           # Tên model embedding (sẽ tạo)
├── thong_ke_du_lieu.png        # Biểu đồ phân tích (sẽ tạo)
└── README.md                    # File hướng dẫn này
```

## 🚀 Cách sử dụng

### 1. Cài đặt thư viện

```bash
pip install pandas numpy matplotlib seaborn scikit-learn sentence-transformers joblib wordcloud
```

### 2. Chạy toàn bộ pipeline

```bash
python spam_detection_complete.py
```

### 3. Sử dụng từng phần riêng lẻ

```python
from spam_detection_complete import SpamDetector

# Tạo detector
detector = SpamDetector()

# Tải dữ liệu
detector.load_data('spam.csv')

# Phân tích dữ liệu
detector.analyze_data()
detector.analyze_keywords()

# Tiền xử lý
detector.preprocess_data()

# Huấn luyện mô hình
accuracy, report = detector.train_model()

# Lưu mô hình
detector.save_model()

# Dự đoán email mới
result = detector.predict_email("FREE MONEY! CLICK HERE!")
print(result)  # "Spam"
```

## 📊 Kết quả

### Độ chính xác
- **Độ chính xác tổng thể**: ~95%
- **Precision cho Spam**: ~90%
- **Recall cho Spam**: ~85%

### Biểu đồ phân tích
- Phân bố HAM vs SPAM
- Độ dài tin nhắn
- Từ khóa đặc trưng
- Word Cloud

## 🔧 Công nghệ sử dụng

- **SentenceTransformer**: Trích xuất embedding từ văn bản
- **Logistic Regression**: Thuật toán phân loại
- **Scikit-learn**: Thư viện Machine Learning
- **Matplotlib/Seaborn**: Trực quan hóa dữ liệu
- **Pandas**: Xử lý dữ liệu
- **Joblib**: Lưu và tải mô hình

## 📈 Quy trình xử lý

1. **Phân tích dữ liệu**: Khám phá và hiểu đặc điểm của email spam/ham
2. **Tiền xử lý**: Làm sạch dữ liệu, tách train/test
3. **Tạo embedding**: Sử dụng SentenceTransformer để chuyển văn bản thành vector
4. **Huấn luyện**: Logistic Regression trên embedding
5. **Đánh giá**: Metrics chi tiết (accuracy, precision, recall, F1)
6. **Lưu mô hình**: Để sử dụng sau

## 🧪 Demo

```python
# Test với các ví dụ
test_emails = [
    "Hello, how are you? I hope you're doing well.",  # Không spam
    "FREE! WIN A PRIZE! CLICK HERE NOW!",             # Spam
    "Meeting tomorrow at 3 PM.",                      # Không spam
    "CONGRATULATIONS! You've won $1000!",             # Spam
]

for email in test_emails:
    result = detector.predict_email(email)
    print(f"Email: {email[:30]}... -> {result}")
```

## 📝 Lưu ý quan trọng

1. **Dữ liệu**: File `spam.csv` phải có định dạng:
   - Cột `v1`: nhãn ('ham' hoặc 'spam')
   - Cột `v2`: nội dung email

2. **Kết nối internet**: Cần để tải SentenceTransformer model lần đầu

3. **Thời gian**: Lần đầu chạy có thể mất 5-10 phút để tải model và tạo embedding

4. **Bộ nhớ**: Cần ít nhất 2GB RAM để chạy mượt mà

## 🚀 Hướng phát triển

1. **Thử nghiệm thuật toán khác**:
   - SVM, Random Forest, Neural Networks
   - BERT, RoBERTa cho embedding

2. **Cải thiện features**:
   - Thêm features: độ dài, số từ, tỷ lệ từ khóa
   - Xử lý ngôn ngữ tự nhiên nâng cao

3. **Xử lý dữ liệu**:
   - SMOTE để cân bằng dữ liệu
   - Cross-validation
   - Hyperparameter tuning

4. **Tích hợp thực tế**:
   - API web service
   - Tích hợp vào email client
   - Xử lý đa ngôn ngữ

## 📞 Hỗ trợ

Nếu gặp vấn đề, hãy kiểm tra:

1. **Lỗi import**: Đảm bảo đã cài đặt đầy đủ thư viện
2. **File dữ liệu**: Kiểm tra `spam.csv` có trong thư mục
3. **Kết nối mạng**: Cần internet để tải model lần đầu
4. **Bộ nhớ**: Đóng các ứng dụng khác nếu thiếu RAM

## 📄 License

Dự án này được tạo cho mục đích học tập và nghiên cứu.

---

**🎉 Chúc bạn thành công với dự án Machine Learning!** 
=======

https://raw.githubusercontent.com/zessycne/BTL_AI/main/spam.csv
>>>>>>> ba4e53c00ef866c17ab816a04cbb1d1cf6c1aab4
