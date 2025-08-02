import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import re
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Đọc dữ liệu
print("📊 KHÁM PHÁ VÀ PHÂN TÍCH DỮ LIỆU EMAIL SPAM")
print("=" * 50)

df = pd.read_csv('spam.csv', encoding='latin-1')
print(f"Tổng số dòng dữ liệu: {len(df)}")

# Thống kê cơ bản
print("\n🔍 THỐNG KÊ CƠ BẢN:")
print("-" * 30)

# Phân loại ham/spam
spam_count = len(df[df['v1'] == 'spam'])
ham_count = len(df[df['v1'] == 'ham'])

print(f"Số lượng email HAM: {ham_count}")
print(f"Số lượng email SPAM: {spam_count}")
print(f"Tỷ lệ HAM: {ham_count/(ham_count+spam_count)*100:.1f}%")
print(f"Tỷ lệ SPAM: {spam_count/(ham_count+spam_count)*100:.1f}%")

# Phân tích độ dài tin nhắn
df['length'] = df['v2'].str.len()
ham_lengths = df[df['v1'] == 'ham']['length']
spam_lengths = df[df['v1'] == 'spam']['length']

print(f"\n📏 PHÂN TÍCH ĐỘ DÀI TIN NHẮN:")
print(f"Độ dài trung bình HAM: {ham_lengths.mean():.1f} ký tự")
print(f"Độ dài trung bình SPAM: {spam_lengths.mean():.1f} ký tự")
print(f"Độ dài tối đa HAM: {ham_lengths.max()} ký tự")
print(f"Độ dài tối đa SPAM: {spam_lengths.max()} ký tự")

# Phân tích từ khóa
print(f"\n🔤 PHÂN TÍCH TỪ KHÓA THƯỜNG XUẤT HIỆN:")

def get_keywords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return [word for word in words if len(word) > 2]

# Từ khóa trong SPAM
spam_texts = ' '.join(df[df['v1'] == 'spam']['v2'].astype(str))
spam_words = get_keywords(spam_texts)
spam_word_freq = Counter(spam_words).most_common(10)

print("Top 10 từ khóa trong SPAM:")
for word, count in spam_word_freq:
    print(f"  - {word}: {count} lần")

# Từ khóa trong HAM
ham_texts = ' '.join(df[df['v1'] == 'ham']['v2'].astype(str))
ham_words = get_keywords(ham_texts)
ham_word_freq = Counter(ham_words).most_common(10)

print("\nTop 10 từ khóa trong HAM:")
for word, count in ham_word_freq:
    print(f"  - {word}: {count} lần")

# Tạo biểu đồ
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
print("6. Việc khám phá dữ liệu giúp chọn phương pháp xử lý phù hợp")

print(f"\n💾 Đã lưu biểu đồ vào file: thong_ke_du_lieu.png") 