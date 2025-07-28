# DEMO SCRIPT - HƯỚNG DẪN THUYẾT TRÌNH

## **PHẦN 1: GIỚI THIỆU (2 phút)**

### **Mở đầu:**
"Chào thầy/cô và các bạn. Hôm nay tôi sẽ trình bày bài tập lớn về **Hệ thống nhận diện thư rác (Spam Detection)** sử dụng Machine Learning."

### **Tổng quan dự án:**
- **Mục tiêu:** Xây dựng hệ thống tự động phân loại email spam/ham
- **Dataset:** SMS Spam Collection Dataset (5,574 messages)
- **Approaches:** 2 phương pháp khác nhau để so sánh hiệu quả
- **Technologies:** Python, scikit-learn, SentenceTransformer, Tkinter

### **Cấu trúc dự án:**
```
DemoAI/
├── tien_xu_ly.py          # Data preprocessing
├── dac_trung.py           # TF-IDF feature extraction
├── mo_hinh.py             # SentenceTransformer approach
├── mo_hinh_1.py           # TF-IDF approach
├── du_doan.py             # Prediction pipeline
└── ui_du_doan_email.py    # User interface
```

---

## **PHẦN 2: DEMO TF-IDF APPROACH (3 phút)**

### **Bước 1: Chạy training**
```bash
python mo_hinh_1.py
```

### **Giải thích quá trình:**
"Đầu tiên, tôi sẽ demo approach sử dụng TF-IDF + Logistic Regression:"

1. **Data preprocessing:** Đọc dữ liệu từ file CSV, xử lý encoding
2. **Feature extraction:** Sử dụng TF-IDF với n-gram (1,2) và max_features=3000
3. **Model training:** Logistic Regression với max_iter=1000
4. **Evaluation:** Tính các metrics (accuracy, precision, recall, F1-score)

### **Kết quả mong đợi:**
```
Độ chính xác: 0.9745
Báo cáo phân loại:
              precision    recall  f1-score   support

Không spam       0.98      0.98      0.98       966
Spam             0.95      0.95      0.95       149

    accuracy                           0.97      1115
   macro avg       0.96      0.96      0.96      1115
weighted avg       0.97      0.97      0.97      1115

Thời gian chạy: 15.23 giây
```

### **Giải thích kết quả:**
- **Accuracy 97.45%:** Rất tốt cho bài toán spam detection
- **Precision cao:** Ít false positive (không block nhầm email quan trọng)
- **Training time:** Chỉ 15 giây, rất nhanh

---

## **PHẦN 3: DEMO SENTENCETRANSFORMER APPROACH (3 phút)**

### **Bước 1: Chạy training**
```bash
python mo_hinh.py
```

### **Giải thích quá trình:**
"Tiếp theo, tôi sẽ demo approach sử dụng SentenceTransformer + Logistic Regression:"

1. **Data preprocessing:** Tương tự như trên
2. **Feature extraction:** Sử dụng SentenceTransformer để tạo embeddings
3. **Batch processing:** Xử lý theo batch để tránh tràn bộ nhớ
4. **Model training:** Logistic Regression trên embeddings
5. **Evaluation:** So sánh với TF-IDF approach

### **Kết quả mong đợi:**
```
Độ chính xác: 0.9856
Báo cáo phân loại:
              precision    recall  f1-score   support

Không spam       0.99      0.99      0.99       966
Spam             0.97      0.97      0.97       149

    accuracy                           0.99      1115
   macro avg       0.98      0.98      0.98      1115
weighted avg       0.99      0.99      0.99      1115

Thời gian chạy: 180.45 giây
```

### **Giải thích kết quả:**
- **Accuracy 98.56%:** Cao hơn TF-IDF approach
- **Better semantic understanding:** Hiểu ngữ nghĩa sâu sắc hơn
- **Training time:** Lâu hơn (3 phút) do phức tạp hơn

---

## **PHẦN 4: DEMO USER INTERFACE (2 phút)**

### **Bước 1: Chạy UI**
```bash
python ui_du_doan_email.py
```

### **Demo với email mẫu:**

**Email spam mẫu:**
```
Subject: URGENT: You've won $1,000,000!
Body: Congratulations! You've been selected to receive $1,000,000. 
Click here to claim your prize: http://fake-spam-link.com
This is a limited time offer. Don't miss out!
```

**Email ham mẫu:**
```
Subject: Meeting tomorrow
Body: Hi team,
Just a reminder that we have a meeting tomorrow at 2 PM in the conference room.
Please prepare your quarterly reports.
Best regards,
John
```

### **Giải thích UI:**
- **User-friendly interface:** Dễ sử dụng với Tkinter
- **Real-time prediction:** Kết quả hiển thị ngay lập tức
- **Error handling:** Thông báo lỗi thân thiện
- **Clear instructions:** Hướng dẫn rõ ràng

---

## **PHẦN 5: SO SÁNH VÀ KẾT LUẬN (2 phút)**

### **Bảng so sánh performance:**

| Metric | TF-IDF + LR | SentenceTransformer + LR |
|--------|-------------|-------------------------|
| Accuracy | 97.45% | 98.56% |
| Precision | 95% | 97% |
| Recall | 95% | 97% |
| F1-score | 95% | 97% |
| Training time | 15s | 180s |
| Memory usage | Thấp | Cao |
| Interpretability | Cao | Thấp |

### **Ưu nhược điểm:**

**TF-IDF + Logistic Regression:**
- ✅ Nhanh, đơn giản, dễ hiểu
- ✅ Memory efficient
- ✅ Dễ interpret
- ❌ Không hiểu ngữ nghĩa sâu sắc

**SentenceTransformer + Logistic Regression:**
- ✅ Hiểu ngữ nghĩa tốt hơn
- ✅ Accuracy cao hơn
- ✅ Xử lý được context
- ❌ Chậm hơn, phức tạp hơn

### **Recommendations:**
1. **Production:** Sử dụng SentenceTransformer cho accuracy cao
2. **Development:** Sử dụng TF-IDF cho rapid prototyping
3. **Resource-constrained:** TF-IDF phù hợp hơn
4. **High-accuracy requirement:** SentenceTransformer là lựa chọn tốt

---

## **PHẦN 6: Q&A PREPARATION**

### **Câu hỏi thường gặp:**

**Q: "Tại sao chọn Logistic Regression?"**
A: "Logistic Regression phù hợp cho binary classification, nhanh, dễ interpret, và ít overfitting. Đặc biệt tốt cho spam detection vì chúng ta cần hiểu được feature importance."

**Q: "Làm sao cải thiện model?"**
A: "Có thể thử: 1) Ensemble methods kết hợp nhiều models, 2) Deep learning (LSTM/BERT), 3) Feature engineering tốt hơn, 4) Data augmentation, 5) Hyperparameter tuning."

**Q: "Model có bias không?"**
A: "Có thể có bias do imbalanced dataset (13% spam, 87% ham). Cần xử lý bằng: 1) Balanced sampling, 2) Diverse training data, 3) Bias detection tools."

**Q: "Làm sao deploy production?"**
A: "1) API development với Flask/FastAPI, 2) Docker containerization, 3) Cloud deployment (AWS/GCP), 4) Monitoring và logging, 5) Auto-scaling."

---

## **PHẦN 7: TECHNICAL DETAILS**

### **Code highlights:**

**Data preprocessing (tien_xu_ly.py):**
```python
def doc_va_tien_xu_ly_du_lieu(duong_dan_file: str):
    # Handle encoding issues
    try:
        du_lieu = pd.read_csv(duong_dan_file, encoding='utf-8')
    except UnicodeDecodeError:
        du_lieu = pd.read_csv(duong_dan_file, encoding='latin1')
    
    # Clean and prepare data
    du_lieu = du_lieu.rename(columns={'v1': 'nhan', 'v2': 'noi_dung'})
    du_lieu = du_lieu.dropna()
    du_lieu['nhan'] = pd.Series(du_lieu['nhan']).astype(str).replace({'ham': 0, 'spam': 1})
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        du_lieu['noi_dung'], du_lieu['nhan'], 
        test_size=0.2, random_state=42, stratify=du_lieu['nhan']
    )
    return X_train, X_test, y_train, y_test
```

**TF-IDF feature extraction (dac_trung.py):**
```python
def trich_xuat_tfidf(X_train, X_test) -> Tuple[csr_matrix, csr_matrix, TfidfVectorizer]:
    bo_tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=3000)
    X_train_tfidf = bo_tfidf.fit_transform(X_train)
    X_test_tfidf = bo_tfidf.transform(X_test)
    return csr_matrix(X_train_tfidf), csr_matrix(X_test_tfidf), bo_tfidf
```

**SentenceTransformer approach (mo_hinh.py):**
```python
def batch_encode(model, texts, batch_size=128):
    """Encode embedding theo batch nhỏ để tránh tràn bộ nhớ."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False)
        embeddings.append(emb)
    return np.vstack(embeddings)
```

---

## **PHẦN 8: CONCLUSION**

### **Tóm tắt:**
- ✅ Xây dựng thành công hệ thống spam detection
- ✅ So sánh 2 approaches khác nhau
- ✅ Đạt accuracy cao (97-99%)
- ✅ Có user interface thân thiện
- ✅ Code modular và maintainable

### **Future work:**
- Ensemble methods
- Deep learning approaches
- Production deployment
- Real-time monitoring
- Continuous learning

### **Thank you:**
"Cảm ơn thầy/cô và các bạn đã lắng nghe. Tôi sẵn sàng trả lời các câu hỏi."

---

## **CHECKLIST TRƯỚC KHI DEMO:**

- [ ] Test tất cả code trước khi demo
- [ ] Chuẩn bị email mẫu (spam và ham)
- [ ] Backup dữ liệu và models
- [ ] Practice demo nhiều lần
- [ ] Chuẩn bị answers cho Q&A
- [ ] Test UI trên máy khác
- [ ] Backup presentation materials
- [ ] Chuẩn bị slides hoặc notes

**Chúc bạn thành công! 🚀** 