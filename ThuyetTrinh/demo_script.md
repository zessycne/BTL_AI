# DEMO SCRIPT - HƯỚNG DẪN THUYẾT TRÌNH

## **PHẦN 1: GIỚI THIỆU (2 phút)**

### **Mở đầu:**
"Chào thầy/cô và các bạn. Hôm nay tôi sẽ trình bày bài tập lớn về **Hệ thống nhận diện thư rác (Spam Detection)** sử dụng Machine Learning."

### **Tổng quan dự án:**
- **Mục tiêu:** Xây dựng hệ thống tự động phân loại email spam/ham
- **Dataset:** SMS Spam Collection Dataset (5,574 messages)
- **Approach:** SentenceTransformer + LogisticRegression với code tối ưu hóa
- **Technologies:** Python, scikit-learn, SentenceTransformer, Tkinter

### **Cấu trúc dự án:**
```
DemoAI/
├── tien_xu_ly.py          # Data preprocessing
├── mo_hinh.py             # Optimized model training (SentenceTransformer)
├── du_doan_email.py       # Command line prediction
├── ui_du_doan_email.py    # User interface
├── spam.csv               # Dataset
├── mo_hinh_spam.pkl       # Trained model
└── sentence_model.txt     # SentenceTransformer model info
```

---

## **PHẦN 2: DEMO SENTENCETRANSFORMER APPROACH (4 phút)**

### **Bước 1: Chạy training**
```bash
python mo_hinh.py
```

### **Giải thích quá trình:**
"Tôi sẽ demo approach sử dụng SentenceTransformer + LogisticRegression với code đã được tối ưu hóa:"

1. **Data preprocessing:** Đọc dữ liệu từ file CSV, xử lý encoding
2. **Text cleaning:** Làm sạch text với hàm `clean_text_list()`
3. **Feature extraction:** Sử dụng SentenceTransformer để tạo embeddings
4. **Batch processing:** Xử lý theo batch để tránh tràn bộ nhớ
5. **Model training:** Logistic Regression với max_iter=1000
6. **Evaluation:** Tính các metrics (accuracy, precision, recall, F1-score)
7. **Model saving:** Lưu mô hình và thông tin embedder

### **Kết quả mong đợi:**
```
=== Huấn luyện mô hình với SentenceTransformer ===
Độ chính xác: 0.9856
Báo cáo phân loại:
              precision    recall  f1-score   support

Không phải rác       0.99      0.99      0.99       966
Thư rác             0.97      0.97      0.97       149

    accuracy                           0.99      1115
   macro avg       0.98      0.98      0.98      1115
weighted avg       0.99      0.99      0.99      1115

Đã lưu mô hình vào mo_hinh_spam.pkl và tên model SentenceTransformer vào sentence_model.txt
```

### **Giải thích kết quả:**
- **Accuracy 98.56%:** Rất tốt cho bài toán spam detection
- **Precision cao:** Ít false positive (không block nhầm email quan trọng)
- **Training time:** Khoảng 3-5 phút, phù hợp với độ phức tạp
- **Code tối ưu:** Đã loại bỏ các hàm trùng lặp, dễ maintain

---

## **PHẦN 3: DEMO USER INTERFACE (3 phút)**

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

## **PHẦN 4: DEMO COMMAND LINE (2 phút)**

### **Bước 1: Chạy command line tool**
```bash
python du_doan_email.py
```

### **Demo với email mẫu:**
```
Nhập email (gõ 'END' để kết thúc):
> Hi, can you send me the meeting notes from yesterday?

Kết quả: Không spam

Nhập email (gõ 'END' để kết thúc):
> FREE VIAGRA NOW!!! Click here to get your free pills!!!

Kết quả: Spam

Nhập email (gõ 'END' để kết thúc):
> END
```

### **Giải thích:**
- **Flexible input:** Hỗ trợ nhập email nhiều dòng
- **Batch processing:** Xử lý hiệu quả với SentenceTransformer
- **Clear output:** Kết quả dễ hiểu

---

## **PHẦN 5: CODE OPTIMIZATION HIGHLIGHTS (2 phút)**

### **Tối ưu hóa đã thực hiện:**

#### **1. Loại bỏ hàm trùng lặp:**
```python
# Đã loại bỏ:
# - encode_sentences() (trùng với batch_encode())
# - xay_dung_va_danh_gia_mo_hinh() (trùng với train_and_evaluate())
```

#### **2. Cấu trúc code rõ ràng:**
```python
# Các hàm chuyên biệt:
def huan_luyen_mo_hinh(X_train_emb, y_train):     # Chỉ huấn luyện
def danh_gia_mo_hinh(mo_hinh, X_test_emb, y_test): # Chỉ đánh giá
def du_doan_tin_nhan(mo_hinh, embedder, tin_nhan): # Chỉ dự đoán
def train_and_evaluate(...):                       # Pipeline chính
```

#### **3. Batch processing hiệu quả:**
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

#### **4. Error handling tốt:**
```python
def clean_text_list(series):
    return [str(s) if pd.notnull(s) and str(s).strip() != "" else "[EMPTY]" for s in series]
```

---

## **PHẦN 6: SO SÁNH VÀ KẾT LUẬN (2 phút)**

### **Ưu điểm của approach hiện tại:**

| Aspect | SentenceTransformer + LR |
|--------|-------------------------|
| Accuracy | 98.56% |
| Precision | 97% |
| Recall | 97% |
| F1-score | 97% |
| Semantic understanding | Cao |
| Code maintainability | Cao |
| Memory efficiency | Tốt (batch processing) |
| Error handling | Tốt |

### **Ưu nhược điểm:**

**SentenceTransformer + Logistic Regression:**
- ✅ Hiểu ngữ nghĩa tốt hơn
- ✅ Accuracy cao (98.56%)
- ✅ Code tối ưu, dễ maintain
- ✅ Batch processing hiệu quả
- ✅ Error handling tốt
- ❌ Training time lâu hơn (3-5 phút)
- ❌ Memory usage cao hơn

### **Recommendations:**
1. **Production:** Sử dụng approach này cho accuracy cao
2. **Development:** Code modular dễ mở rộng
3. **Maintenance:** Code sạch, ít trùng lặp
4. **Performance:** Batch processing tối ưu

---

## **PHẦN 7: Q&A PREPARATION**

### **Câu hỏi thường gặp:**

**Q: "Tại sao chọn SentenceTransformer?"**
A: "SentenceTransformer hiểu ngữ nghĩa sâu sắc hơn TF-IDF, phù hợp cho việc phân loại email spam vì có thể hiểu context và ý nghĩa thực sự của tin nhắn."

**Q: "Làm sao cải thiện model?"**
A: "Có thể thử: 1) Ensemble methods kết hợp nhiều models, 2) Deep learning (LSTM/BERT), 3) Feature engineering tốt hơn, 4) Data augmentation, 5) Hyperparameter tuning."

**Q: "Code có tối ưu không?"**
A: "Đã tối ưu bằng cách: 1) Loại bỏ hàm trùng lặp, 2) Batch processing, 3) Modular design, 4) Error handling tốt, 5) Memory management hiệu quả."

**Q: "Làm sao deploy production?"**
A: "1) API development với Flask/FastAPI, 2) Docker containerization, 3) Cloud deployment (AWS/GCP), 4) Monitoring và logging, 5) Auto-scaling."

---

## **PHẦN 8: TECHNICAL DETAILS**

### **Code highlights:**

**Optimized training pipeline (mo_hinh.py):**
```python
def train_and_evaluate(duong_dan_file: str, duong_dan_mo_hinh: str, duong_dan_embedder: str):
    """Pipeline: train, test, lưu mô hình và tên model embedding."""
    # Đọc và tiền xử lý dữ liệu
    X_train, X_test, y_train, y_test = doc_va_tien_xu_ly_du_lieu(duong_dan_file)
    
    # Khởi tạo SentenceTransformer
    embedder = SentenceTransformer(MODEL_NAME)
    
    # Tiền xử lý và encode dữ liệu
    X_train_clean = clean_text_list(X_train)
    X_test_clean = clean_text_list(X_test)
    X_train_emb = batch_encode(embedder, X_train_clean)
    X_test_emb = batch_encode(embedder, X_test_clean)
    
    # Huấn luyện mô hình
    mo_hinh = huan_luyen_mo_hinh(X_train_emb, y_train)
    
    # Đánh giá mô hình
    do_chinh_xac, bao_cao = danh_gia_mo_hinh(mo_hinh, X_test_emb, y_test)
    
    # Lưu mô hình
    luu_mo_hinh_va_embedder(mo_hinh, duong_dan_mo_hinh, duong_dan_embedder)
    
    return mo_hinh, embedder
```

**Efficient batch processing:**
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

**Modular prediction function:**
```python
def du_doan_tin_nhan(mo_hinh, embedder, tin_nhan: str):
    """Dự đoán một tin nhắn/email là spam hay không spam."""
    tin_nhan_clean = clean_text_list([tin_nhan])
    tin_nhan_emb = batch_encode(embedder, tin_nhan_clean)
    du_doan = mo_hinh.predict(tin_nhan_emb)[0]
    return "Spam" if du_doan == 1 else "Không spam"
```

---

## **PHẦN 9: CONCLUSION**

### **Tóm tắt:**
- ✅ Xây dựng thành công hệ thống spam detection với accuracy 98.56%
- ✅ Code tối ưu hóa, loại bỏ trùng lặp, dễ maintain
- ✅ Batch processing hiệu quả, tránh tràn bộ nhớ
- ✅ User interface thân thiện
- ✅ Error handling tốt
- ✅ Modular design cho dễ mở rộng

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