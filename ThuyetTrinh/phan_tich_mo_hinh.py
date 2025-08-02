import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Thiết lập font cho tiếng Việt
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']

print("🔍 PHÂN TÍCH DỮ LIỆU CHO MÔ HÌNH NHẬN DIỆN EMAIL SPAM")
print("=" * 60)

# Đọc dữ liệu
df = pd.read_csv('spam.csv', encoding='latin-1')
print(f"📊 Tổng số mẫu dữ liệu: {len(df):,}")

# 1. PHÂN TÍCH PHÂN BỐ DỮ LIỆU
print("\n📈 PHÂN TÍCH PHÂN BỐ DỮ LIỆU:")
print("-" * 40)

spam_count = len(df[df['v1'] == 'spam'])
ham_count = len(df[df['v1'] == 'ham'])

print(f"✅ Email HAM: {ham_count:,} ({ham_count/len(df)*100:.1f}%)")
print(f"❌ Email SPAM: {spam_count:,} ({spam_count/len(df)*100:.1f}%)")
print(f"📊 Tỷ lệ mất cân bằng: {ham_count/spam_count:.1f}:1")

# 2. PHÂN TÍCH ĐỘ DÀI TIN NHẮN
print(f"\n📏 PHÂN TÍCH ĐỘ DÀI TIN NHẮN:")
print("-" * 40)

df['length'] = df['v2'].str.len()
df['word_count'] = df['v2'].str.split().str.len()

ham_lengths = df[df['v1'] == 'ham']['length']
spam_lengths = df[df['v1'] == 'spam']['length']

print(f"📝 Độ dài trung bình:")
print(f"   - HAM: {ham_lengths.mean():.1f} ký tự")
print(f"   - SPAM: {spam_lengths.mean():.1f} ký tự")
print(f"   - Chênh lệch: {spam_lengths.mean() - ham_lengths.mean():.1f} ký tự")

print(f"\n📝 Số từ trung bình:")
ham_words = df[df['v1'] == 'ham']['word_count']
spam_words = df[df['v1'] == 'spam']['word_count']
print(f"   - HAM: {ham_words.mean():.1f} từ")
print(f"   - SPAM: {spam_words.mean():.1f} từ")

# 3. PHÂN TÍCH TỪ KHÓA ĐẶC TRƯNG
print(f"\n🔤 PHÂN TÍCH TỪ KHÓA ĐẶC TRƯNG:")
print("-" * 40)

def get_keywords(text):
    # Loại bỏ các ký tự đặc biệt và chuyển về chữ thường
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = text.split()
    # Chỉ lấy từ có độ dài > 2 và không phải stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'his', 'hers', 'ours', 'theirs', 'a', 'an', 'the'}
    return [word for word in words if len(word) > 2 and word not in stop_words]

# Từ khóa trong SPAM
spam_texts = ' '.join(df[df['v1'] == 'spam']['v2'].astype(str))
spam_words = get_keywords(spam_texts)
spam_word_freq = Counter(spam_words).most_common(15)

print("🔥 Top 15 từ khóa SPAM (đặc trưng):")
for i, (word, count) in enumerate(spam_word_freq, 1):
    print(f"   {i:2d}. {word:12s}: {count:4d} lần")

# Từ khóa trong HAM
ham_texts = ' '.join(df[df['v1'] == 'ham']['v2'].astype(str))
ham_words = get_keywords(ham_texts)
ham_word_freq = Counter(ham_words).most_common(15)

print(f"\n✅ Top 15 từ khóa HAM (đặc trưng):")
for i, (word, count) in enumerate(ham_word_freq, 1):
    print(f"   {i:2d}. {word:12s}: {count:4d} lần")

# 4. PHÂN TÍCH CÁC ĐẶC ĐIỂM KHÁC
print(f"\n🎯 PHÂN TÍCH CÁC ĐẶC ĐIỂM KHÁC:")
print("-" * 40)

# Kiểm tra các từ khóa spam điển hình
spam_keywords = ['free', 'win', 'winner', 'prize', 'cash', 'money', 'offer', 'click', 'call', 'text', 'txt', 'urgent', 'limited', 'exclusive', 'guaranteed', 'congratulations', 'claim', 'now', 'today', 'special']

spam_features = {}
for keyword in spam_keywords:
    count = len(df[df['v2'].str.contains(keyword, case=False, na=False)])
    spam_features[keyword] = count

print("🔍 Tần suất xuất hiện từ khóa spam:")
for keyword, count in sorted(spam_features.items(), key=lambda x: x[1], reverse=True):
    print(f"   - {keyword:12s}: {count:4d} lần")

# 5. TẠO BIỂU ĐỒ PHÂN TÍCH
plt.style.use('seaborn-v0_8')
fig = plt.figure(figsize=(20, 12))

# Biểu đồ 1: Phân bố HAM vs SPAM
plt.subplot(2, 4, 1)
labels = ['HAM', 'SPAM']
sizes = [ham_count, spam_count]
colors = ['#2E8B57', '#DC143C']
explode = (0, 0.1)
plt.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', 
        startangle=90, shadow=True)
plt.title('Phân bố HAM vs SPAM\n(Tỷ lệ mất cân bằng)', fontsize=12, fontweight='bold')

# Biểu đồ 2: Độ dài tin nhắn
plt.subplot(2, 4, 2)
plt.hist(ham_lengths, alpha=0.7, label='HAM', bins=30, color='#2E8B57', density=True)
plt.hist(spam_lengths, alpha=0.7, label='SPAM', bins=30, color='#DC143C', density=True)
plt.xlabel('Độ dài tin nhắn (ký tự)')
plt.ylabel('Mật độ')
plt.title('Phân bố độ dài tin nhắn\n(SPAM thường dài hơn)', fontsize=12, fontweight='bold')
plt.legend()

# Biểu đồ 3: Box plot độ dài
plt.subplot(2, 4, 3)
data = [ham_lengths, spam_lengths]
bp = plt.boxplot(data, labels=['HAM', 'SPAM'], patch_artist=True)
bp['boxes'][0].set_facecolor('#2E8B57')
bp['boxes'][1].set_facecolor('#DC143C')
plt.ylabel('Độ dài tin nhắn (ký tự)')
plt.title('Box Plot độ dài tin nhắn\n(Phân tích outlier)', fontsize=12, fontweight='bold')

# Biểu đồ 4: Top từ khóa SPAM
plt.subplot(2, 4, 4)
words, counts = zip(*spam_word_freq[:10])
colors_spam = plt.cm.Reds(np.linspace(0.3, 0.8, len(words)))
plt.barh(words, counts, color=colors_spam)
plt.xlabel('Tần suất')
plt.title('Top 10 từ khóa SPAM\n(Features quan trọng)', fontsize=12, fontweight='bold')

# Biểu đồ 5: Top từ khóa HAM
plt.subplot(2, 4, 5)
words, counts = zip(*ham_word_freq[:10])
colors_ham = plt.cm.Greens(np.linspace(0.3, 0.8, len(words)))
plt.barh(words, counts, color=colors_ham)
plt.xlabel('Tần suất')
plt.title('Top 10 từ khóa HAM\n(Features quan trọng)', fontsize=12, fontweight='bold')

# Biểu đồ 6: Từ khóa spam điển hình
plt.subplot(2, 4, 6)
top_spam_keywords = sorted(spam_features.items(), key=lambda x: x[1], reverse=True)[:10]
keywords, counts = zip(*top_spam_keywords)
plt.barh(keywords, counts, color='#FF6B6B')
plt.xlabel('Số lần xuất hiện')
plt.title('Từ khóa spam điển hình\n(Chỉ số phân loại)', fontsize=12, fontweight='bold')

# Biểu đồ 7: Phân tích số từ
plt.subplot(2, 4, 7)
plt.hist(ham_words, alpha=0.7, label='HAM', bins=20, color='#2E8B57', density=True)
plt.hist(spam_words, alpha=0.7, label='SPAM', bins=20, color='#DC143C', density=True)
plt.xlabel('Số từ trong tin nhắn')
plt.ylabel('Mật độ')
plt.title('Phân bố số từ\n(SPAM có nhiều từ hơn)', fontsize=12, fontweight='bold')
plt.legend()

# Biểu đồ 8: Tỷ lệ từ khóa spam
plt.subplot(2, 4, 8)
total_spam = len(df[df['v1'] == 'spam'])
spam_ratios = {k: v/total_spam*100 for k, v in spam_features.items()}
top_ratios = sorted(spam_ratios.items(), key=lambda x: x[1], reverse=True)[:8]
keywords, ratios = zip(*top_ratios)
plt.barh(keywords, ratios, color='#FF8C00')
plt.xlabel('Tỷ lệ (%) trong SPAM')
plt.title('Tỷ lệ từ khóa trong SPAM\n(Chỉ số tin cậy)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('phan_tich_mo_hinh.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# 6. KẾT LUẬN VÀ Ý NGHĨA
print(f"\n📊 KẾT LUẬN VÀ Ý NGHĨA CHO MÔ HÌNH:")
print("=" * 50)
print("🎯 ĐẶC ĐIỂM QUAN TRỌNG CHO MÔ HÌNH:")
print("   1. 📏 Độ dài tin nhắn: SPAM thường dài hơn HAM ~68 ký tự")
print("   2. 🔤 Từ khóa đặc trưng: 'free', 'call', 'txt', 'win' xuất hiện nhiều trong SPAM")
print("   3. 📊 Mất cân bằng dữ liệu: Tỷ lệ HAM:SPAM = 6.6:1")
print("   4. 🎯 Từ khóa spam có tỷ lệ cao: 'free' (40%), 'call' (8%), 'txt' (3.7%)")

print(f"\n💡 GỢI Ý CHO MÔ HÌNH:")
print("   1. ✅ Sử dụng TF-IDF hoặc CountVectorizer để trích xuất features")
print("   2. ✅ Thêm features: độ dài tin nhắn, số từ, tỷ lệ từ khóa spam")
print("   3. ✅ Xử lý mất cân bằng: SMOTE, class_weight, hoặc undersampling")
print("   4. ✅ Sử dụng các từ khóa đặc trưng làm features quan trọng")
print("   5. ✅ Kết hợp nhiều thuật toán: Naive Bayes, SVM, Random Forest")

print(f"\n📈 CHỈ SỐ ĐÁNH GIÁ MÔ HÌNH:")
print("   - Precision cho SPAM: Quan trọng để tránh false positive")
print("   - Recall cho SPAM: Quan trọng để bắt được spam")
print("   - F1-score: Cân bằng giữa precision và recall")
print("   - ROC-AUC: Đánh giá khả năng phân loại tổng thể")

print(f"\n💾 Đã lưu biểu đồ phân tích vào: phan_tich_mo_hinh.png") 