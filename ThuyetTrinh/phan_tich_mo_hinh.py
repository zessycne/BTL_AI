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

# Đọc dữ liệu
df = pd.read_csv('spam.csv', encoding='latin-1')

# 1. PHÂN TÍCH PHÂN BỐ DỮ LIỆU
spam_count = len(df[df['v1'] == 'spam'])
ham_count = len(df[df['v1'] == 'ham'])

# 2. PHÂN TÍCH ĐỘ DÀI TIN NHẮN
df['length'] = df['v2'].str.len()
df['word_count'] = df['v2'].str.split().str.len()

ham_lengths = df[df['v1'] == 'ham']['length']
spam_lengths = df[df['v1'] == 'spam']['length']

ham_words = df[df['v1'] == 'ham']['word_count']
spam_words = df[df['v1'] == 'spam']['word_count']

# 3. PHÂN TÍCH TỪ KHÓA ĐẶC TRƯNG
def get_keywords(text):
    # Loại bỏ các ký tự đặc biệt và chuyển về chữ thường
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = text.split()
    # Chỉ lấy từ có độ dài > 2 và không phải stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'his', 'hers', 'ours', 'theirs', 'a', 'an', 'the'}
    return [word for word in words if len(word) > 2 and word not in stop_words]

# Từ khóa trong SPAM
spam_texts = ' '.join(df[df['v1'] == 'spam']['v2'].astype(str))
spam_words_list = get_keywords(spam_texts)
spam_word_freq = Counter(spam_words_list).most_common(15)

# Từ khóa trong HAM
ham_texts = ' '.join(df[df['v1'] == 'ham']['v2'].astype(str))
ham_words_list = get_keywords(ham_texts)
ham_word_freq = Counter(ham_words_list).most_common(15)

# 4. PHÂN TÍCH CÁC ĐẶC ĐIỂM KHÁC
# Kiểm tra các từ khóa spam điển hình
spam_keywords = ['free', 'win', 'winner', 'prize', 'cash', 'money', 'offer', 'click', 'call', 'text', 'txt', 'urgent', 'limited', 'exclusive', 'guaranteed', 'congratulations', 'claim', 'now', 'today', 'special']

spam_features = {}
for keyword in spam_keywords:
    count = len(df[df['v2'].str.contains(keyword, case=False, na=False)])
    spam_features[keyword] = count

# 5. TẠO BIỂU ĐỒ PHÂN TÍCH
plt.style.use('seaborn-v0_8')
fig = plt.figure(figsize=(28, 16))

# Biểu đồ 1: Phân bố HAM vs SPAM
plt.subplot(2, 4, 1)
labels = ['HAM', 'SPAM']
sizes = [ham_count, spam_count]
colors = ['#2E8B57', '#DC143C']
explode = (0, 0.1)
plt.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', 
        startangle=90, shadow=True, textprops={'fontsize': 9})
plt.title('Phân bố HAM vs SPAM\n(Tỷ lệ mất cân bằng)', fontsize=10, fontweight='bold', pad=25)

# Biểu đồ 2: Độ dài tin nhắn
plt.subplot(2, 4, 2)
plt.hist(ham_lengths, alpha=0.7, label='HAM', bins=30, color='#2E8B57', density=True)
plt.hist(spam_lengths, alpha=0.7, label='SPAM', bins=30, color='#DC143C', density=True)
plt.xlabel('Độ dài tin nhắn (ký tự)', fontsize=9)
plt.ylabel('Mật độ', fontsize=9)
plt.title('Phân bố độ dài tin nhắn\n(SPAM thường dài hơn)', fontsize=10, fontweight='bold', pad=25)
plt.legend(fontsize=8)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

# Biểu đồ 3: Box plot độ dài
plt.subplot(2, 4, 3)
data = [ham_lengths, spam_lengths]
bp = plt.boxplot(data, labels=['HAM', 'SPAM'], patch_artist=True)
bp['boxes'][0].set_facecolor('#2E8B57')
bp['boxes'][1].set_facecolor('#DC143C')
plt.ylabel('Độ dài tin nhắn (ký tự)', fontsize=9)
plt.title('Box Plot độ dài tin nhắn\n(Phân tích outlier)', fontsize=10, fontweight='bold', pad=25)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

# Biểu đồ 4: Top từ khóa SPAM
plt.subplot(2, 4, 4)
words, counts = zip(*spam_word_freq[:10])
colors_spam = plt.cm.Reds(np.linspace(0.3, 0.8, len(words)))
bars = plt.barh(words, counts, color=colors_spam)
plt.xlabel('Tần suất', fontsize=9)
plt.title('Top 10 từ khóa SPAM\n(Features quan trọng)', fontsize=10, fontweight='bold', pad=25)
plt.xticks(fontsize=8)
plt.yticks(fontsize=7)

# Biểu đồ 5: Top từ khóa HAM
plt.subplot(2, 4, 5)
words, counts = zip(*ham_word_freq[:10])
colors_ham = plt.cm.Greens(np.linspace(0.3, 0.8, len(words)))
bars = plt.barh(words, counts, color=colors_ham)
plt.xlabel('Tần suất', fontsize=9)
plt.title('Top 10 từ khóa HAM\n(Features quan trọng)', fontsize=10, fontweight='bold', pad=25)
plt.xticks(fontsize=8)
plt.yticks(fontsize=7)

# Biểu đồ 6: Từ khóa spam điển hình
plt.subplot(2, 4, 6)
top_spam_keywords = sorted(spam_features.items(), key=lambda x: x[1], reverse=True)[:10]
keywords, counts = zip(*top_spam_keywords)
bars = plt.barh(keywords, counts, color='#FF6B6B')
plt.xlabel('Số lần xuất hiện', fontsize=9)
plt.title('Từ khóa spam điển hình\n(Chỉ số phân loại)', fontsize=10, fontweight='bold', pad=25)
plt.xticks(fontsize=8)
plt.yticks(fontsize=7)

# Biểu đồ 7: Phân tích số từ
plt.subplot(2, 4, 7)
plt.hist(ham_words, alpha=0.7, label='HAM', bins=20, color='#2E8B57', density=True)
plt.hist(spam_words, alpha=0.7, label='SPAM', bins=20, color='#DC143C', density=True)
plt.xlabel('Số từ trong tin nhắn', fontsize=9)
plt.ylabel('Mật độ', fontsize=9)
plt.title('Phân bố số từ\n(SPAM có nhiều từ hơn)', fontsize=10, fontweight='bold', pad=25)
plt.legend(fontsize=8)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

# Biểu đồ 8: Tỷ lệ từ khóa spam
plt.subplot(2, 4, 8)
total_spam = len(df[df['v1'] == 'spam'])
spam_ratios = {k: v/total_spam*100 for k, v in spam_features.items()}
top_ratios = sorted(spam_ratios.items(), key=lambda x: x[1], reverse=True)[:8]
keywords, ratios = zip(*top_ratios)
bars = plt.barh(keywords, ratios, color='#FF8C00')
plt.xlabel('Tỷ lệ (%) trong SPAM', fontsize=9)
plt.title('Tỷ lệ từ khóa trong SPAM\n(Chỉ số tin cậy)', fontsize=10, fontweight='bold', pad=25)
plt.xticks(fontsize=8)
plt.yticks(fontsize=7)

plt.tight_layout(pad=4.0, h_pad=2.0, w_pad=2.0)
plt.show() 