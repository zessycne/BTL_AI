import tkinter as tk
from tkinter import scrolledtext, messagebox
from du_doan import tai_mo_hinh_va_vectorizer, du_doan_tin_nhan

# Tải mô hình và vectorizer
mo_hinh, vectorizer = tai_mo_hinh_va_vectorizer('mo_hinh_spam.pkl', 'vectorizer_spam.pkl')

def du_doan_email():
    email = text_email.get('1.0', tk.END).strip()
    if not email:
        messagebox.showwarning('Cảnh báo', 'Vui lòng nhập nội dung email!')
        return
    ket_qua = du_doan_tin_nhan(mo_hinh, vectorizer, email)
    label_ket_qua.config(text=f'Kết quả: {ket_qua}')

root = tk.Tk()
root.title('Nhận diện Email Spam')
root.geometry('500x400')

label_huongdan = tk.Label(root, text='Nhập nội dung email cần kiểm tra:', font=('Arial', 12))
label_huongdan.pack(pady=10)

text_email = scrolledtext.ScrolledText(root, width=60, height=12, font=('Arial', 11))
text_email.pack(padx=10, pady=5)

btn_du_doan = tk.Button(root, text='Dự đoán', font=('Arial', 12, 'bold'), command=du_doan_email)
btn_du_doan.pack(pady=10)

label_ket_qua = tk.Label(root, text='Kết quả: ', font=('Arial', 12, 'bold'))
label_ket_qua.pack(pady=10)

root.mainloop() 