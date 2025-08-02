# 📊 BÁO CÁO PHÂN TÍCH DỮ LIỆU EMAIL SPAM

## 📈 Tổng quan dữ liệu
- **Tổng số mẫu**: 5,572 email
- **Email HAM**: 4,825 (86.6%)
- **Email SPAM**: 747 (13.4%)
- **Tỷ lệ mất cân bằng**: 6.5:1 (HAM:SPAM)

## 🔍 Đặc điểm quan trọng cho mô hình

### 1. 📏 Độ dài tin nhắn
- **HAM trung bình**: 71.0 ký tự
- **SPAM trung bình**: 138.9 ký tự
- **Chênh lệch**: 67.9 ký tự
- **Kết luận**: SPAM thường dài hơn HAM gần gấp đôi

### 2. 🔤 Từ khóa đặc trưng SPAM
| Thứ tự | Từ khóa | Tần suất | Ý nghĩa |
|--------|---------|----------|---------|
| 1 | call | 355 lần | Kêu gọi hành động |
| 2 | free | 224 lần | Quảng cáo miễn phí |
| 3 | now | 199 lần | Tính khẩn cấp |
| 4 | your | 264 lần | Cá nhân hóa |
| 5 | txt | 163 lần | Hướng dẫn SMS |
| 6 | claim | 113 lần | Yêu cầu nhận thưởng |
| 7 | prize | 93 lần | Giải thưởng |
| 8 | win | 166 lần | Chiến thắng |

### 3. ✅ Từ khóa đặc trưng HAM
| Thứ tự | Từ khóa | Tần suất | Ý nghĩa |
|--------|---------|----------|---------|
| 1 | not | 415 lần | Phủ định |
| 2 | get | 305 lần | Hành động thường ngày |
| 3 | how | 304 lần | Câu hỏi |
| 4 | now | 300 lần | Thời gian |
| 5 | just | 293 lần | Chỉ, vừa |
| 6 | when | 287 lần | Thời gian |
| 7 | what | 273 lần | Câu hỏi |
| 8 | call | 236 lần | Gọi điện |

## 🎯 Tỷ lệ từ khóa spam trong toàn bộ dữ liệu
- **now**: 732 lần (13.1%)
- **call**: 635 lần (11.4%)
- **free**: 265 lần (4.8%)
- **text**: 224 lần (4.0%)
- **txt**: 194 lần (3.5%)
- **today**: 180 lần (3.2%)
- **win**: 166 lần (3.0%)

## 💡 Gợi ý cho mô hình

### 1. 🔧 Xử lý dữ liệu
- **Vectorization**: Sử dụng TF-IDF hoặc CountVectorizer
- **Feature engineering**: Thêm độ dài tin nhắn, số từ
- **Cân bằng dữ liệu**: SMOTE hoặc class_weight

### 2. 🎯 Features quan trọng
- **Từ khóa spam**: free, call, txt, win, prize, claim
- **Độ dài tin nhắn**: SPAM > HAM
- **Số từ**: SPAM có nhiều từ hơn
- **Tỷ lệ từ khóa spam**: Chỉ số tin cậy

### 3. 📊 Thuật toán đề xuất
1. **Naive Bayes**: Hiệu quả với text classification
2. **SVM**: Xử lý tốt dữ liệu không cân bằng
3. **Random Forest**: Ít bị overfitting
4. **Logistic Regression**: Baseline model

### 4. 📈 Chỉ số đánh giá
- **Precision cho SPAM**: Quan trọng để tránh false positive
- **Recall cho SPAM**: Quan trọng để bắt được spam
- **F1-score**: Cân bằng giữa precision và recall
- **ROC-AUC**: Đánh giá tổng thể

## 🚀 Kết luận

### Đặc điểm quan trọng:
1. **Mất cân bằng dữ liệu**: Cần xử lý đặc biệt
2. **Độ dài tin nhắn**: Feature quan trọng
3. **Từ khóa đặc trưng**: Chỉ số phân loại mạnh
4. **Tỷ lệ từ khóa**: Chỉ số tin cậy cao

### Hướng phát triển:
1. **Feature engineering**: Kết hợp nhiều đặc điểm
2. **Ensemble methods**: Kết hợp nhiều mô hình
3. **Hyperparameter tuning**: Tối ưu hóa tham số
4. **Cross-validation**: Đánh giá ổn định

---
*Báo cáo được tạo tự động từ phân tích dữ liệu spam.csv* 