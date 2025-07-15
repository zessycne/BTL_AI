from mo_hinh import tai_mo_hinh, clean_text_list, batch_encode

def du_doan_tin_nhan(mo_hinh, embedder, tin_nhan: str):
    """Dự đoán một email là spam hay không spam."""
    tin_nhan_clean = clean_text_list([tin_nhan])
    tin_nhan_emb = batch_encode(embedder, tin_nhan_clean)
    du_doan = mo_hinh.predict(tin_nhan_emb)[0]
    return "Spam" if du_doan == 1 else "Không spam"

if __name__ == "__main__":
    mo_hinh, embedder = tai_mo_hinh('mo_hinh_spam.pkl', 'sentence_model.txt')
    print("Nhập nội dung email cần kiểm tra:")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    email = "\n".join(lines)
    ket_qua = du_doan_tin_nhan(mo_hinh, embedder, email)
    print("Kết quả dự đoán:", ket_qua) 