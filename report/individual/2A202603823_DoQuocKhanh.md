# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603823
- **Họ và tên:** Đỗ Quốc Khánh
- **SEED** (in ở ô đầu notebook): 1327

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | x |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | x |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | x |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*
                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.832      0.908        0.974   
2. BoW + MLP (deep learning)         0.978   0.901      0.938        1.000   
3a. Encoder (MiniLM) + kNN           0.885   0.990      0.935        1.000   
3b. Encoder (MiniLM) + LogReg        0.952   0.990      0.971        1.000   
3c. Encoder (MiniLM) + MLP head      0.961   0.970      0.966        1.000   
4a. LLM nhỏ zero-shot                0.950   0.950      0.950        0.950   
4c. LLM nhỏ few-shot                 0.952   1.000      0.976        0.976   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            3.629      44216.635       0.050   
2. BoW + MLP (deep learning)         5.730      43940.231       0.169   
3a. Encoder (MiniLM) + kNN          89.811         20.022      56.114   
3b. Encoder (MiniLM) + LogReg       90.278         29.092      31.482   
3c. Encoder (MiniLM) + MLP head     90.806         28.599      30.134   
4a. LLM nhỏ zero-shot                0.000          0.741     784.398   
4c. LLM nhỏ few-shot                 0.000          0.959     713.725   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.088         0.000  
2. BoW + MLP (deep learning)          0.278         0.000  
3a. Encoder (MiniLM) + kNN           79.044         0.000  
3b. Encoder (MiniLM) + LogReg        48.687         0.000  
3c. Encoder (MiniLM) + MLP head      42.501         0.000  
4a. LLM nhỏ zero-shot              3457.119         0.018  
4c. LLM nhỏ few-shot               2523.024         0.049  

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*

Em đoán F1 của BoW + LogReg khoảng 0.85, nhưng kết quả là 0.908. Model làm tốt hơn em nghĩ vì các từ như “free”, “call”, “win” xuất hiện khá rõ trong tin spam.

Em đoán lr=50 sẽ làm loss tăng mạnh hoặc không ổn định. Thực tế loss chỉ tăng ở đầu rồi giảm nhanh về mức thấp. Như vậy learning rate lớn không phải lúc nào cũng làm model bị lỗi, nhưng vẫn có rủi ro nếu dùng với dữ liệu khác.

Em đoán MLP có thể bằng hoặc kém Logistic Regression. Kết quả MLP có F1 = 0.938, cao hơn LogReg là 0.908. Tuy nhiên mức tăng không quá lớn và MLP train lâu hơn.
## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

Model BoW + LogReg tụt nhiều khi gặp các case khó, vì model này chỉ nhìn các từ riêng lẻ. Ví dụ case viết biến dạng như C0ngr4tulat10ns hoặc w1n-gift.c0m làm các từ không còn giống từ trong vocab. Một ví dụ khác là tin nhắn nói về spam như “I won £1000…” nhưng thực chất đang cảnh báo lừa đảo; BoW dễ thấy các từ won, £1000 rồi đoán nhầm là spam.

LLM có prompt phòng thủ làm tốt hơn ở nhóm prompt injection. Kết quả model 4d. LLM nhỏ + prompt phòng thủ đúng 50/50 case, cho thấy việc coi nội dung SMS là dữ liệu thay vì mệnh lệnh đã có tác dụng.
## 5. Thử thách cá nhân

Em thử vocab 100 và 5000 từ với Logistic Regression.
- Vocab 100 từ: F1 test = 0.827, train = 0.34 giây.
- Vocab 5000 từ: F1 test = 0.908, train = 18.64 giây.
Vocab 100 làm F1 giảm vì mất nhiều từ đặc trưng của spam. Vocab 5000 giữ được nhiều thông tin hơn nên F1 tốt hơn, nhưng thời gian train tăng nhiều. Vì vậy không phải vocab càng lớn càng tốt, mà cần cân bằng giữa chất lượng và tốc độ.

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*
Đề #2: ML · Giảm vocab còn 100 từ và tăng lên 5000 từ (build_vocab(..., max_size)). Train lại, báo F1 test. Giải thích xu hướng.

code: 
challenge_results = {}

for vocab_size in [100, 5000]:
    vocab_test = build_vocab(texts_tr, max_size=vocab_size)

    X_tr_test = vectorize(texts_tr, vocab_test)
    X_te_test = vectorize(texts_te, vocab_test)

    t0 = time.perf_counter()
    w_test, b_test, _ = train_logreg(
        X_tr_test, y_tr,
        lr=0.5,
        epochs=300
    )
    train_s = time.perf_counter() - t0

    pred_test = (
        predict_proba(X_te_test, w_test, b_test) >= 0.5
    ).astype(int)

    m_test = metrics(y_te, pred_test)

    challenge_results[vocab_size] = {
        "f1": m_test["f1"],
        "train_s": train_s,
        "so_tu": len(vocab_test)
    }

    print(
        f"Vocab {vocab_size} "
        f"(thực tế {len(vocab_test)} từ): "
        f"F1 test = {m_test['f1']:.3f} · "
        f"train = {train_s:.2f}s"
    )
Đáp án: 
Vocab 100 (thực tế 100 từ): F1 test = 0.827 · train = 0.34s
Vocab 5000 (thực tế 5000 từ): F1 test = 0.908 · train = 18.64s

Giải thích: Vocab 100 từ cho F1 thấp hơn vì bị mất nhiều từ đặc trưng của spam. Vocab 5000 từ thường cho F1 cao hơn vì giữ được nhiều thông tin hơn, nhưng thời gian train cũng tăng.
## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

Với yêu cầu p95 dưới 50 ms, các cách đạt yêu cầu là BoW + LogReg (0.088 ms), BoW + MLP (0.278 ms), Encoder + LogReg (48.687 ms) và Encoder + MLP head (42.501 ms).

Em chọn BoW + MLP vì F1 test là 0.938, chạy rất nhanh và không mất chi phí API. Nếu cần F1 cao hơn thì Encoder + LogReg có F1 = 0.971, nhưng gần sát giới hạn SLA hơn và cần thời gian tạo embedding.

LLM không phù hợp để lọc toàn bộ 10 triệu SMS/ngày vì p95 rất cao: zero-shot là 3457 ms, few-shot là 2523 ms. Với giá zero-shot khoảng $0.018/1k tin, chi phí ước tính là khoảng $5,400/tháng; few-shot khoảng $14,700/tháng.

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
| Phần | Bạn hỏi AI gì | AI trả lời/viết gì | Bạn đã kiểm tra |
|---|---|---|---|
| Phần LLM | Hỏi lỗi quota Gemini và cấu hình `reasoning_effort` | Giải thích lỗi 429 do hết quota API và cần dùng `low` thay cho `minimal` ở Gemini 3.8 | Chạy lại model nhỏ và kiểm tra output |
| Phần báo cáo | Hỏi cách giải thích kết quả F1, Pareto và stress test | Gợi ý cách diễn đạt ngắn, dễ hiểu | Đối chiếu lại với bảng kết quả của notebook |
| Thử thách vocab | Hỏi cấu trúc code để thử vocab 100 và 5000 | Gợi ý vòng lặp tạo vocab, vector hóa và train lại | Chạy code, lấy F1 = 0.827 và 0.908 |

Qua bài này em hiểu rằng F1 cao chưa đủ để chọn model. Cần xem thêm độ trễ, chi phí và dữ liệu thực tế. Em cũng hiểu rằng LLM tiện khi chưa có dữ liệu train, nhưng không phải lúc nào cũng phù hợp để chạy số lượng lớn.