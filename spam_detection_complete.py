#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📧 HỆ THỐNG NHẬN DIỆN EMAIL SPAM
======================================

Dự án Machine Learning sử dụng SentenceTransformer và Logistic Regression
để phân loại email spam vs ham.

Tác giả: DemoAI
Ngày tạo: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer
import joblib
from collections import Counter
import re
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Thiết lập font cho tiếng Việt
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.style.use('seaborn-v0_8')

class SpamDetector:
    """Lớp chính để xây dựng và sử dụng mô hình nhận diện spam."""
    
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        self.model_name = model_name
        self.embedder = None
        self.model = None
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, file_path='spam.csv'):
        """Đọc và tải dữ liệu từ file CSV."""
        print("📊 KHÁM PHÁ VÀ PHÂN TÍCH DỮ LIỆU EMAIL SPAM")
        print("=" * 50)
        
        try:
            self.df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            self.df = pd.read_csv(file_path, encoding='latin-1')
            
        print(f"Tổng số dòng dữ liệu: {len(self.df)}")
        return self.df
    
    def analyze_data(self):
        """Phân tích dữ liệu cơ bản."""
        if self.df is None:
            print("❌ Chưa tải dữ liệu. Hãy gọi load_data() trước.")
            return
            
        print("\n🔍 THỐNG KÊ CƠ BẢN:")
        print("-" * 30)
        
        # Phân loại ham/spam
        spam_count = len(self.df[self.df['v1'] == 'spam'])
        ham_count = len(self.df[self.df['v1'] == 'ham'])
        
        print(f"Số lượng email HAM: {ham_count}")
        print(f"Số lượng email SPAM: {spam_count}")
        print(f"Tỷ lệ HAM: {ham_count/(ham_count+spam_count)*100:.1f}%")
        print(f"Tỷ lệ SPAM: {spam_count/(ham_count+spam_count)*100:.1f}%")
        
        # Phân tích độ dài tin nhắn
        self.df['length'] = self.df['v2'].str.len()
        ham_lengths = self.df[self.df['v1'] == 'ham']['length']
        spam_lengths = self.df[self.df['v1'] == 'spam']['length']
        
        print(f"\n📏 PHÂN TÍCH ĐỘ DÀI TIN NHẮN:")
        print(f"Độ dài trung bình HAM: {ham_lengths.mean():.1f} ký tự")
        print(f"Độ dài trung bình SPAM: {spam_lengths.mean():.1f} ký tự")
        print(f"Độ dài tối đa HAM: {ham_lengths.max()} ký tự")
        print(f"Độ dài tối đa SPAM: {spam_lengths.max()} ký tự")
        
        return ham_count, spam_count, ham_lengths, spam_lengths
    
    def analyze_keywords(self):
        """Phân tích từ khóa trong dữ liệu."""
        print(f"\n🔤 PHÂN TÍCH TỪ KHÓA THƯỜNG XUẤT HIỆN:")
        
        def get_keywords(text):
            words = re.findall(r'\b\w+\b', text.lower())
            return [word for word in words if len(word) > 2]
        
        # Từ khóa trong SPAM
        spam_texts = ' '.join(self.df[self.df['v1'] == 'spam']['v2'].astype(str))
        spam_words = get_keywords(spam_texts)
        spam_word_freq = Counter(spam_words).most_common(10)
        
        print("Top 10 từ khóa trong SPAM:")
        for word, count in spam_word_freq:
            print(f"  - {word}: {count} lần")
        
        # Từ khóa trong HAM
        ham_texts = ' '.join(self.df[self.df['v1'] == 'ham']['v2'].astype(str))
        ham_words = get_keywords(ham_texts)
        ham_word_freq = Counter(ham_words).most_common(10)
        
        print("\nTop 10 từ khóa trong HAM:")
        for word, count in ham_word_freq:
            print(f"  - {word}: {count} lần")
            
        return spam_word_freq, ham_word_freq
    
    def create_visualizations(self, ham_count, spam_count, ham_lengths, spam_lengths, 
                            spam_word_freq, ham_word_freq):
        """Tạo biểu đồ phân tích dữ liệu."""
        plt.figure(figsize=(15, 10))
        
        # Biểu đồ phân loại
        plt.subplot(2, 3, 1)
        labels = ['HAM', 'SPAM']
        sizes = [ham_count, spam_count]
        colors = ['#66b3ff', '#ff9999']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Phân bố HAM vs SPAM')
        
        # Biểu đồ độ dài
        plt.subplot(2, 3, 2)
        plt.hist(ham_lengths, alpha=0.7, label='HAM', bins=30, color='blue')
        plt.hist(spam_lengths, alpha=0.7, label='SPAM', bins=30, color='red')
        plt.xlabel('Độ dài tin nhắn')
        plt.ylabel('Tần suất')
        plt.title('Phân bố độ dài tin nhắn')
        plt.legend()
        
        # Box plot độ dài
        plt.subplot(2, 3, 3)
        data = [ham_lengths, spam_lengths]
        plt.boxplot(data, labels=['HAM', 'SPAM'])
        plt.ylabel('Độ dài tin nhắn')
        plt.title('Box Plot độ dài tin nhắn')
        
        # Từ khóa SPAM
        plt.subplot(2, 3, 4)
        words, counts = zip(*spam_word_freq[:8])
        plt.barh(words, counts, color='red', alpha=0.7)
        plt.xlabel('Tần suất')
        plt.title('Top từ khóa SPAM')
        
        # Từ khóa HAM
        plt.subplot(2, 3, 5)
        words, counts = zip(*ham_word_freq[:8])
        plt.barh(words, counts, color='blue', alpha=0.7)
        plt.xlabel('Tần suất')
        plt.title('Top từ khóa HAM')
        
        # Word Cloud SPAM
        plt.subplot(2, 3, 6)
        spam_texts = ' '.join(self.df[self.df['v1'] == 'spam']['v2'].astype(str))
        wordcloud = WordCloud(width=400, height=200, background_color='white', 
                             colormap='Reds', max_words=50).generate(spam_texts)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud SPAM')
        
        plt.tight_layout()
        plt.savefig('thong_ke_du_lieu.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📈 Ý NGHĨA VÀ ẢNH HƯỞNG:")
        print("-" * 30)
        print("1. Dữ liệu có sự mất cân bằng nhẹ giữa HAM và SPAM")
        print("2. Tin nhắn SPAM thường dài hơn và chứa nhiều từ khóa quảng cáo")
        print("3. Các từ khóa như 'free', 'win', 'click', 'offer' xuất hiện nhiều trong SPAM")
        print("4. Dữ liệu này giúp hiểu rõ hơn về đặc điểm của email spam")
        print("5. Có thể sử dụng các từ khóa này làm features cho mô hình")
    
    def preprocess_data(self):
        """Tiền xử lý dữ liệu cho mô hình."""
        print("🔧 TIỀN XỬ LÝ DỮ LIỆU")
        print("=" * 30)
        
        # Đổi tên cột cho dễ xử lý
        du_lieu = self.df.rename(columns={'v1': 'nhan', 'v2': 'noi_dung'})
        
        # Chỉ giữ 2 cột cần thiết
        du_lieu = du_lieu[['nhan', 'noi_dung']]
        
        # Loại bỏ dòng bị thiếu dữ liệu
        du_lieu = du_lieu.dropna()
        
        # Đảm bảo cột 'nhan' là Series, dùng .replace đúng chuẩn
        du_lieu['nhan'] = pd.Series(du_lieu['nhan']).astype(str).replace({'ham': 0, 'spam': 1})
        
        # Tách tập train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            du_lieu['noi_dung'], du_lieu['nhan'], test_size=0.2, random_state=42, stratify=du_lieu['nhan']
        )
        
        print(f"Kích thước tập train: {len(self.X_train)} mẫu")
        print(f"Kích thước tập test: {len(self.X_test)} mẫu")
        print(f"Tỷ lệ spam trong train: {self.y_train.mean():.3f}")
        print(f"Tỷ lệ spam trong test: {self.y_test.mean():.3f}")
        
        # Hiển thị một số mẫu
        print("\n📝 MỘT SỐ MẪU DỮ LIỆU:")
        for i in range(3):
            print(f"\nMẫu {i+1}:")
            print(f"Nội dung: {self.X_train.iloc[i][:100]}...")
            print(f"Nhãn: {'SPAM' if self.y_train.iloc[i] == 1 else 'HAM'}")
    
    def clean_text_list(self, series):
        """Làm sạch dữ liệu đầu vào."""
        return [str(s) if pd.notnull(s) and str(s).strip() != "" else "[EMPTY]" for s in series]
    
    def batch_encode(self, texts, batch_size=128):
        """Encode embedding theo batch nhỏ để tránh tràn bộ nhớ."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            emb = self.embedder.encode(batch, show_progress_bar=False)
            embeddings.append(emb)
        return np.vstack(embeddings)
    
    def train_model(self):
        """Huấn luyện mô hình."""
        print("🤖 XÂY DỰNG MÔ HÌNH VỚI SENTENCETRANSFORMER")
        print("=" * 50)
        
        # Tải SentenceTransformer model
        print("📥 Đang tải SentenceTransformer model...")
        self.embedder = SentenceTransformer(self.model_name)
        print(f"✅ Đã tải model: {self.model_name}")
        
        # Tiền xử lý và tạo embedding
        print("\n🔄 Đang tạo embedding cho dữ liệu...")
        X_train_clean = self.clean_text_list(self.X_train)
        X_test_clean = self.clean_text_list(self.X_test)
        
        print("   - Đang encode tập train...")
        X_train_emb = self.batch_encode(X_train_clean)
        print("   - Đang encode tập test...")
        X_test_emb = self.batch_encode(X_test_clean)
        
        print(f"✅ Hoàn thành! Kích thước embedding: {X_train_emb.shape[1]} chiều")
        
        # Huấn luyện mô hình
        print("\n🎯 HUẤN LUYỆN MÔ HÌNH LOGISTIC REGRESSION")
        print("=" * 45)
        
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X_train_emb, self.y_train)
        print("✅ Hoàn thành huấn luyện!")
        
        # Đánh giá mô hình
        print("\n📊 ĐÁNH GIÁ MÔ HÌNH")
        print("=" * 20)
        
        y_pred = self.model.predict(X_test_emb)
        do_chinh_xac = accuracy_score(self.y_test, y_pred)
        bao_cao = classification_report(self.y_test, y_pred, target_names=['Không phải rác', 'Thư rác'])
        
        print(f"Độ chính xác tổng thể: {do_chinh_xac:.4f}")
        print("\nBáo cáo phân loại chi tiết:")
        print(bao_cao)
        
        return do_chinh_xac, bao_cao
    
    def predict_email(self, email_text):
        """Dự đoán một email là spam hay không spam."""
        if self.model is None or self.embedder is None:
            print("❌ Mô hình chưa được huấn luyện. Hãy gọi train_model() trước.")
            return None
            
        email_clean = self.clean_text_list([email_text])
        email_emb = self.batch_encode(email_clean)
        prediction = self.model.predict(email_emb)[0]
        return "Spam" if prediction == 1 else "Không spam"
    
    def save_model(self, model_path='mo_hinh_spam.pkl', embedder_path='sentence_model.txt'):
        """Lưu mô hình và tên model embedding vào file."""
        if self.model is None:
            print("❌ Mô hình chưa được huấn luyện. Hãy gọi train_model() trước.")
            return
            
        print("💾 LƯU MÔ HÌNH")
        print("=" * 15)
        
        joblib.dump(self.model, model_path)
        with open(embedder_path, 'w', encoding='utf-8') as f:
            f.write(self.model_name)
            
        print(f"✅ Đã lưu mô hình vào '{model_path}'")
        print(f"✅ Đã lưu tên model vào '{embedder_path}'")
    
    def load_saved_model(self, model_path='mo_hinh_spam.pkl', embedder_path='sentence_model.txt'):
        """Tải mô hình đã lưu từ file."""
        print("📥 TẢI MÔ HÌNH ĐÃ LƯU")
        print("=" * 25)
        
        try:
            self.model = joblib.load(model_path)
            with open(embedder_path, 'r', encoding='utf-8') as f:
                model_name = f.read().strip()
            self.embedder = SentenceTransformer(model_name)
            print("✅ Đã tải mô hình thành công!")
            return True
        except Exception as e:
            print(f"❌ Lỗi khi tải mô hình: {e}")
            return False
    
    def demo_predictions(self):
        """Demo dự đoán với các ví dụ."""
        print("🧪 DEMO DỰ ĐOÁN EMAIL SPAM")
        print("=" * 30)
        
        test_emails = [
            "Hello, how are you? I hope you're doing well.",
            "FREE! WIN A PRIZE! CLICK HERE NOW! LIMITED TIME OFFER!",
            "Meeting tomorrow at 3 PM. Please bring the documents.",
            "CONGRATULATIONS! You've won $1000! Claim your prize now!",
            "Hi mom, I'll be home late tonight. Love you!"
        ]
        
        print("📧 KẾT QUẢ DỰ ĐOÁN:")
        print("-" * 25)
        
        for i, email in enumerate(test_emails, 1):
            ket_qua = self.predict_email(email)
            print(f"\n{i}. Email: {email[:50]}...")
            print(f"   Kết quả: {ket_qua}")
    
    def run_complete_pipeline(self, data_path='spam.csv'):
        """Chạy toàn bộ pipeline từ đầu đến cuối."""
        print("🚀 CHẠY TOÀN BỘ PIPELINE SPAM DETECTION")
        print("=" * 50)
        
        # 1. Tải dữ liệu
        self.load_data(data_path)
        
        # 2. Phân tích dữ liệu
        ham_count, spam_count, ham_lengths, spam_lengths = self.analyze_data()
        spam_word_freq, ham_word_freq = self.analyze_keywords()
        
        # 3. Tạo biểu đồ
        self.create_visualizations(ham_count, spam_count, ham_lengths, spam_lengths,
                                 spam_word_freq, ham_word_freq)
        
        # 4. Tiền xử lý
        self.preprocess_data()
        
        # 5. Huấn luyện mô hình
        accuracy, report = self.train_model()
        
        # 6. Lưu mô hình
        self.save_model()
        
        # 7. Demo
        self.demo_predictions()
        
        print("\n🎉 HOÀN THÀNH PIPELINE!")
        print(f"Độ chính xác cuối cùng: {accuracy:.4f}")
        
        return accuracy, report


def main():
    """Hàm chính để chạy dự án."""
    print("📧 HỆ THỐNG NHẬN DIỆN EMAIL SPAM")
    print("=" * 50)
    print("Tác giả: DemoAI")
    print("Sử dụng SentenceTransformer + Logistic Regression")
    print()
    
    # Tạo instance của SpamDetector
    detector = SpamDetector()
    
    # Chạy toàn bộ pipeline
    try:
        accuracy, report = detector.run_complete_pipeline()
        
        print("\n" + "="*50)
        print("📋 TÓM TẮT KẾT QUẢ:")
        print(f"✅ Độ chính xác: {accuracy:.4f}")
        print("✅ Mô hình đã được lưu vào 'mo_hinh_spam.pkl'")
        print("✅ Biểu đồ đã được lưu vào 'thong_ke_du_lieu.png'")
        print("✅ Có thể sử dụng detector.predict_email() để dự đoán email mới")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        print("💡 Hãy đảm bảo file spam.csv có trong thư mục hiện tại")


if __name__ == "__main__":
    main() 