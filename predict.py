import pandas as pd
from mo_hinh import tai_mo_hinh, batch_encode
from tien_xu_ly import clean_text_list

def du_doan_file_csv(duong_dan_file: str, duong_dan_mo_hinh: str, duong_dan_embedder: str):
    """Dự đoán spam cho tất cả email trong file CSV."""
    # Tải mô hình và embedder
    print("Đang tải mô hình...")
    mo_hinh, embedder = tai_mo_hinh(duong_dan_mo_hinh, duong_dan_embedder)
    
    # Đọc file CSV
    print(f"Đang đọc file {duong_dan_file}...")
    df = pd.read_csv(duong_dan_file)
    
    # Kiểm tra cột dữ liệu
    if 'v2' not in df.columns:
        print("Lỗi: Không tìm thấy cột 'v2' trong file CSV")
        return
    
    # Lấy danh sách email
    emails = df['v2'].tolist()
    print(f"Tìm thấy {len(emails)} email cần dự đoán")
    
    # Tiền xử lý dữ liệu
    print("Đang tiền xử lý dữ liệu...")
    emails_clean = clean_text_list(emails)
    
    # Encode embedding
    print("Đang tạo embedding...")
    emails_emb = batch_encode(embedder, emails_clean)
    
    # Dự đoán
    print("Đang thực hiện dự đoán...")
    du_doan = mo_hinh.predict(emails_emb)
    
    # Tạo kết quả
    ket_qua = []
    for i, (email, pred) in enumerate(zip(emails, du_doan)):
        label = "Spam" if pred == 1 else "Không spam"
        ket_qua.append({
            'STT': i + 1,
            'Email': email,
            'Dự đoán': label,
            'Nhãn số': pred
        })
    
    # Tạo DataFrame kết quả
    df_ket_qua = pd.DataFrame(ket_qua)
    
    # Hiển thị kết quả
    print("\n" + "="*80)
    print("KẾT QUẢ DỰ ĐOÁN SPAM")
    print("="*80)
    
    for _, row in df_ket_qua.iterrows():
        print(f"\nSTT {row['STT']}:")
        print(f"Email: {row['Email'][:100]}{'...' if len(row['Email']) > 100 else ''}")
        print(f"Dự đoán: {row['Dự đoán']}")
        print("-" * 80)
    
    # Thống kê
    spam_count = len(df_ket_qua[df_ket_qua['Dự đoán'] == 'Spam'])
    non_spam_count = len(df_ket_qua[df_ket_qua['Dự đoán'] == 'Không spam'])
    
    print(f"\nTHỐNG KÊ:")
    print(f"Tổng số email: {len(df_ket_qua)}")
    print(f"Số email spam: {spam_count}")
    print(f"Số email không spam: {non_spam_count}")
    print(f"Tỷ lệ spam: {spam_count/len(df_ket_qua)*100:.2f}%")
    
    # Lưu kết quả ra file
    ten_file_ket_qua = 'ket_qua_du_doan.csv'
    df_ket_qua.to_csv(ten_file_ket_qua, index=False, encoding='utf-8-sig')
    print(f"\nĐã lưu kết quả vào file: {ten_file_ket_qua}")
    
    return df_ket_qua

def du_doan_tin_nhan(mo_hinh, embedder, tin_nhan: str):
    """Dự đoán một email là spam hay không spam."""
    tin_nhan_clean = clean_text_list([tin_nhan])
    tin_nhan_emb = batch_encode(embedder, tin_nhan_clean)
    du_doan = mo_hinh.predict(tin_nhan_emb)[0]
    return "Spam" if du_doan == 1 else "Không spam"

if __name__ == "__main__":
    print("=== DỰ ĐOÁN SPAM THỰC TẾ ===")
    
    # Dự đoán từ file CSV
    try:
        ket_qua = du_doan_file_csv('predict_thucte.csv', 'mo_hinh_spam.pkl', 'sentence_model.txt')
        print("\nHoàn thành dự đoán!")
        
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file - {e}")
    except Exception as e:
        print(f"Lỗi: {e}")
        print("\nThử dự đoán từng email một:")
        
        # Fallback: dự đoán từng email một
        try:
            mo_hinh, embedder = tai_mo_hinh('mo_hinh_spam.pkl', 'sentence_model.txt')
            df = pd.read_csv('predict_thucte.csv')
            
            for i, email in enumerate(df['v2']):
                ket_qua = du_doan_tin_nhan(mo_hinh, embedder, email)
                print(f"Email {i+1}: {ket_qua}")
                
        except Exception as e2:
            print(f"Lỗi fallback: {e2}")
