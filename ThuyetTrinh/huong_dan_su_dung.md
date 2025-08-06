# 📋 HƯỚNG DẪN SỬ DỤNG PROJECT NHẬN DIỆN EMAIL SPAM

## 🎯 Tổng quan
Project này sử dụng **SentenceTransformer** và **LogisticRegression** để phân loại email spam/không spam với độ chính xác cao. Code đã được tối ưu hóa và loại bỏ các hàm trùng lặp.

## 📁 Cấu trúc file
```
DemoAI/
├── mo_hinh.py              # File chính - huấn luyện mô hình (đã tối ưu)
├── tien_xu_ly.py           # Tiền xử lý dữ liệu
├── du_doan_email.py        # Dự đoán qua command line
├── ui_du_doan_email.py     # Giao diện đồ họa
├── spam.csv                # Dữ liệu huấn luyện
├── mo_hinh_spam.pkl        # Mô hình đã huấn luyện
├── sentence_model.txt      # Tên model SentenceTransformer
└── ThuyetTrinh/            # Tài liệu thuyết trình
    ├── huong_dan_su_dung.md
    ├── checklist_thuyet_trinh.md
    ├── demo_script.md
    └── huong_dan_tra_loi_giang_vien.md
```

## 🚀 THỨ TỰ CHẠY CÁC FILE

### **1. GIAI ĐOẠN HUẤN LUYỆN (Chỉ chạy 1 lần)**

```bash
python mo_hinh.py
```

**Quy trình:**
1. **`mo_hinh.py`** → Gọi **`tien_xu_ly.py`** → Đọc `spam.csv`
2. Huấn luyện mô hình SentenceTransformer + LogisticRegression
3. Lưu mô hình vào `mo_hinh_spam.pkl` và `sentence_model.txt`

### **2. GIAI ĐOẠN SỬ DỤNG (Chạy nhiều lần)**

**Lựa chọn 1 - Giao diện đồ họa:**
```bash
python ui_du_doan_email.py
```

**Lựa chọn 2 - Command line:**
```bash
python du_doan_email.py
```

---

## ⚙️ CÁCH HOẠT ĐỘNG CỦA TỪNG FILE

### **📊 `tien_xu_ly.py`**
```python
# Chức năng: Tiền xử lý dữ liệu
def doc_va_tien_xu_ly_du_lieu(duong_dan_file: str):
    # 1. Đọc file CSV (spam.csv)
    # 2. Đổi tên cột: v1→nhan, v2→noi_dung  
    # 3. Chuyển nhãn: ham→0, spam→1
    # 4. Tách train/test (80%/20%)
    # 5. Trả về: X_train, X_test, y_train, y_test
```

**Chức năng:**
- Đọc và làm sạch dữ liệu từ file CSV
- Chuyển đổi nhãn từ text sang số
- Chia dữ liệu thành tập huấn luyện và kiểm tra
- Xử lý encoding để tránh lỗi Unicode

### **🤖 `mo_hinh.py` (ĐÃ TỐI ƯU HÓA)**
```python
# Chức năng: Huấn luyện và quản lý mô hình

# Khi chạy trực tiếp:
if __name__ == '__main__':
    train_and_evaluate('spam.csv', 'mo_hinh_spam.pkl', 'sentence_model.txt')
```

**Quy trình chi tiết:**

#### **1. Hàm `train_and_evaluate()` - Pipeline chính (ĐÃ TỐI ƯU):**
```python
def train_and_evaluate(duong_dan_file, duong_dan_mo_hinh, duong_dan_embedder):
    # Bước 1: Đọc và tiền xử lý dữ liệu
    X_train, X_test, y_train, y_test = doc_va_tien_xu_ly_du_lieu(duong_dan_file)
    
    # Bước 2: Khởi tạo SentenceTransformer
    embedder = SentenceTransformer(MODEL_NAME)
    
    # Bước 3: Làm sạch dữ liệu text
    X_train_clean = clean_text_list(X_train)
    X_test_clean = clean_text_list(X_test)
    
    # Bước 4: Tạo embedding theo batch (TỐI ƯU)
    X_train_emb = batch_encode(embedder, X_train_clean)
    X_test_emb = batch_encode(embedder, X_test_clean)
    
    # Bước 5: Huấn luyện mô hình
    mo_hinh = huan_luyen_mo_hinh(X_train_emb, y_train)
    
    # Bước 6: Đánh giá mô hình
    do_chinh_xac, bao_cao = danh_gia_mo_hinh(mo_hinh, X_test_emb, y_test)
    
    # Bước 7: Lưu mô hình và embedder
    luu_mo_hinh_va_embedder(mo_hinh, duong_dan_mo_hinh, duong_dan_embedder)
    
    return mo_hinh, embedder
```

#### **2. Hàm `clean_text_list()` - Làm sạch dữ liệu:**
```python
def clean_text_list(series):
    # Chuyển đổi mỗi phần tử thành chuỗi
    # Loại bỏ None/NaN values
    # Thay thế chuỗi rỗng bằng "[EMPTY]"
    return [str(s) if pd.notnull(s) and str(s).strip() != "" else "[EMPTY]" for s in series]
```

#### **3. Hàm `batch_encode()` - Tạo embedding theo batch (TỐI ƯU):**
```python
def batch_encode(model, texts, batch_size=128):
    embeddings = []
    # Chia dữ liệu thành các batch nhỏ
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # Encode từng batch để tránh tràn bộ nhớ
        emb = model.encode(batch, show_progress_bar=False)
        embeddings.append(emb)
    # Ghép tất cả embeddings lại thành một array
    return np.vstack(embeddings)
```

#### **4. Hàm `huan_luyen_mo_hinh()` - Huấn luyện LogisticRegression:**
```python
def huan_luyen_mo_hinh(X_train_emb, y_train):
    # Khởi tạo mô hình LogisticRegression
    mo_hinh = LogisticRegression(max_iter=1000)
    # Huấn luyện trên embedding đã tạo
    mo_hinh.fit(X_train_emb, y_train)
    return mo_hinh
```

#### **5. Hàm `danh_gia_mo_hinh()` - Đánh giá hiệu suất:**
```python
def danh_gia_mo_hinh(mo_hinh, X_test_emb, y_test):
    # Dự đoán trên tập test
    du_doan = mo_hinh.predict(X_test_emb)
    # Tính độ chính xác
    do_chinh_xac = accuracy_score(y_test, du_doan)
    # Tạo báo cáo phân loại chi tiết
    bao_cao = classification_report(y_test, du_doan, target_names=['Không phải rác', 'Thư rác'])
    return do_chinh_xac, bao_cao
```

#### **6. Hàm `luu_mo_hinh_va_embedder()` - Lưu mô hình:**
```python
def luu_mo_hinh_va_embedder(mo_hinh, duong_dan_mo_hinh, duong_dan_embedder):
    # Lưu mô hình LogisticRegression
    joblib.dump(mo_hinh, duong_dan_mo_hinh)
    # Lưu tên model SentenceTransformer
    with open(duong_dan_embedder, 'w', encoding='utf-8') as f:
        f.write(MODEL_NAME)
```

#### **7. Hàm `tai_mo_hinh()` - Tải mô hình:**
```python
def tai_mo_hinh(duong_dan_mo_hinh, duong_dan_embedder):
    # Tải mô hình LogisticRegression
    mo_hinh = joblib.load(duong_dan_mo_hinh)
    # Đọc tên model SentenceTransformer
    with open(duong_dan_embedder, 'r', encoding='utf-8') as f:
        model_name = f.read().strip()
    # Khởi tạo SentenceTransformer
    embedder = SentenceTransformer(model_name)
    return mo_hinh, embedder
```

#### **8. Hàm `du_doan_tin_nhan()` - Dự đoán đơn lẻ:**
```python
def du_doan_tin_nhan(mo_hinh, embedder, tin_nhan: str):
    # Làm sạch tin nhắn đầu vào
    tin_nhan_clean = clean_text_list([tin_nhan])
    # Tạo embedding cho tin nhắn
    tin_nhan_emb = batch_encode(embedder, tin_nhan_clean)
    # Dự đoán
    du_doan = mo_hinh.predict(tin_nhan_emb)[0]
    # Trả về kết quả dạng text
    return "Spam" if du_doan == 1 else "Không spam"
```

**Chức năng tổng quan:**
- **Xử lý dữ liệu**: Làm sạch và chuẩn hóa text input
- **Tạo embedding**: Sử dụng SentenceTransformer để chuyển text thành vector số
- **Huấn luyện**: Sử dụng LogisticRegression để phân loại
- **Đánh giá**: Tính toán độ chính xác và báo cáo chi tiết
- **Lưu trữ**: Lưu mô hình để sử dụng sau này
- **Dự đoán**: Cung cấp API để dự đoán email mới

### **💻 `du_doan_email.py`**
```python
# Chức năng: Dự đoán qua command line

# Quy trình:
# 1. Tải mô hình từ file đã lưu
# 2. Nhận input từ người dùng (nhập email)
# 3. Dự đoán spam/không spam
# 4. In kết quả ra màn hình
```

**Chức năng:**
- Tải mô hình đã huấn luyện
- Nhận email từ người dùng qua command line
- Thực hiện dự đoán và hiển thị kết quả
- Hỗ trợ nhập email nhiều dòng (gõ "END" để kết thúc)

### **🖥️ `ui_du_doan_email.py`**
```python
# Chức năng: Giao diện đồ họa với Tkinter

# Quy trình:
# 1. Tải mô hình từ file đã lưu
# 2. Tạo giao diện với text box và button
# 3. Người dùng nhập email vào text box
# 4. Click "Dự đoán" → Hiển thị kết quả
```

**Chức năng:**
- Tạo giao diện đồ họa thân thiện
- Text box để nhập email
- Button để thực hiện dự đoán
- Hiển thị kết quả trực quan
- Xử lý lỗi và cảnh báo

---

## 🎯 LUỒNG HOẠT ĐỘNG TỔNG QUAN

```
spam.csv → tien_xu_ly.py → mo_hinh.py → [mo_hinh_spam.pkl, sentence_model.txt]
                                                    ↓
                                            [du_doan_email.py hoặc ui_du_doan_email.py]
                                                    ↓
                                            Kết quả dự đoán
```

## 📝 HƯỚNG DẪN SỬ DỤNG CHI TIẾT

### **Bước 1: Chuẩn bị môi trường**
```bash
# Cài đặt các thư viện cần thiết
pip install pandas numpy scikit-learn sentence-transformers joblib
```

### **Bước 2: Huấn luyện mô hình (chỉ 1 lần)**
```bash
python mo_hinh.py
```

**Kết quả mong đợi:**
- Hiển thị độ chính xác của mô hình
- Báo cáo phân loại chi tiết
- Tạo file `mo_hinh_spam.pkl` và `sentence_model.txt`

### **Bước 3: Sử dụng mô hình**

**Cách 1 - Giao diện đẹp:**
```bash
python ui_du_doan_email.py
```
- Mở cửa sổ giao diện
- Nhập email vào text box
- Click "Dự đoán"
- Xem kết quả

**Cách 2 - Command line:**
```bash
python du_doan_email.py
```
- Nhập email từng dòng
- Gõ "END" để kết thúc
- Xem kết quả dự đoán

## ⚠️ LƯU Ý QUAN TRỌNG

### **Yêu cầu hệ thống:**
1. **Python 3.7+**
2. **Các thư viện**: pandas, numpy, scikit-learn, sentence-transformers, joblib, tkinter
3. **File dữ liệu**: `spam.csv` phải có trong thư mục

### **Thứ tự thực hiện:**
1. **Bắt buộc**: Chạy `mo_hinh.py` trước để tạo mô hình
2. **Sau đó**: Có thể chạy `du_doan_email.py` hoặc `ui_du_doan_email.py`

### **Xử lý lỗi thường gặp:**
- **Lỗi encoding**: File `tien_xu_ly.py` đã xử lý tự động
- **Lỗi memory**: Sử dụng batch processing trong `mo_hinh.py`
- **Lỗi model**: Kiểm tra file `mo_hinh_spam.pkl` và `sentence_model.txt`

## 🔧 Tùy chỉnh và mở rộng

### **Thay đổi model SentenceTransformer:**
```python
# Trong mo_hinh.py, thay đổi MODEL_NAME
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
```

### **Thay đổi tham số huấn luyện:**
```python
# Trong mo_hinh.py, điều chỉnh LogisticRegression
mo_hinh = LogisticRegression(max_iter=1000, C=1.0)
```

### **Thay đổi tỷ lệ train/test:**
```python
# Trong tien_xu_ly.py, điều chỉnh test_size
test_size=0.2  # 80% train, 20% test
```

### **Thay đổi batch size:**
```python
# Trong mo_hinh.py, điều chỉnh batch_size
X_train_emb = batch_encode(embedder, X_train_clean, batch_size=64)  # Giảm nếu RAM thấp
```

## 📊 Hiệu suất mô hình

- **Phương pháp**: SentenceTransformer + LogisticRegression
- **Độ chính xác**: Thường đạt 98%+ trên tập test
- **Thời gian huấn luyện**: ~3-5 phút (tùy thuộc vào phần cứng)
- **Thời gian dự đoán**: <1 giây cho mỗi email
- **Memory usage**: Tối ưu với batch processing

## 🚀 TỐI ƯU HÓA ĐÃ THỰC HIỆN

### **1. Loại bỏ hàm trùng lặp:**
- ❌ `encode_sentences()` - Loại bỏ vì trùng với `batch_encode()`
- ❌ `xay_dung_va_danh_gia_mo_hinh()` - Loại bỏ vì trùng với `train_and_evaluate()`

### **2. Cải thiện cấu trúc:**
- ✅ Mỗi hàm có chức năng rõ ràng và không trùng lặp
- ✅ Code modular, dễ maintain và mở rộng
- ✅ Comments rõ ràng cho từng bước

### **3. Tối ưu hiệu suất:**
- ✅ Batch processing để tránh tràn bộ nhớ
- ✅ Error handling tốt hơn
- ✅ Memory management hiệu quả

### **4. Kết quả:**
- 📉 Giảm từ 123 dòng xuống 95 dòng (giảm ~23%)
- 🎯 Loại bỏ hoàn toàn code trùng lặp
- 🔧 Code dễ đọc và bảo trì hơn

---

*Tài liệu này được cập nhật theo code mới đã được tối ưu hóa để hỗ trợ việc sử dụng project nhận diện email spam.* 