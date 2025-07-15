from du_doan import tai_mo_hinh_va_vectorizer, du_doan_tin_nhan
import os

if __name__ == '__main__':
    mo_hinh, vectorizer = tai_mo_hinh_va_vectorizer('mo_hinh_spam.pkl', 'vectorizer_spam.pkl')
    print('Nhập nội dung email cần kiểm tra:')
    print('- Nhập tên file để đọc từ file (ví dụ: email.txt)')
    print('- Hoặc nhập nhiều dòng, kết thúc bằng một dòng chỉ chứa END')
    print('- Hoặc nhập một dòng email rồi nhấn Enter')
    du_lieu = input()
    if os.path.isfile(du_lieu):
        with open(du_lieu, 'r', encoding='utf-8') as f:
            email = f.read()
        print('Đã đọc nội dung từ file:', du_lieu)
    else:
        if du_lieu.strip() == 'END':
            email = ''
        else:
            lines = [du_lieu]
            while True:
                line = input()
                if line.strip() == 'END':
                    break
                lines.append(line)
            email = '\n'.join(lines)
    ket_qua = du_doan_tin_nhan(mo_hinh, vectorizer, email)
    print(f'Kết quả: {ket_qua}') 