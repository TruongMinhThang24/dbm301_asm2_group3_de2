# Bảng đối chiếu Rubric và bằng chứng cần nộp

**Môn:** DBM301 – Assignment 2 – Đề 2 (Groceries)  
**Nhóm:** Group 3  
**Mục đích:** Dùng bảng này làm checklist nghiệm thu. Đây là yêu cầu và bằng chứng cần chuẩn bị, không phải xác nhận các phân tích đã hoàn thành.

## Quy ước đầu ra

| Tên trong bảng | Vị trí/tên khi nộp |
|---|---|
| Notebook | `notebook.ipynb` — một notebook duy nhất, có Phần 1–6, chạy Restart & Run All trong tối đa 15 phút |
| Báo cáo | `baocao.docx` theo mẫu; tối đa 12 trang phần chính, phụ lục được tính riêng |
| Slide | `slide.pptx` — tối đa 15 slide |
| Dự đoán | `predictions.csv` — 1.535 dòng, đúng hai cột theo thứ tự `basket_id,prob`; `prob` từ 0 đến 1 |
| Điểm bất thường | `anomaly_scores.csv` — 1.535 dòng, đúng hai cột theo thứ tự `basket_id,score`; điểm lớn hơn nghĩa là bất thường hơn |
| Môi trường | `requirements.txt` — thư viện và phiên bản thực sự dùng |
| Phụ lục | Phụ lục A: nhật ký AI; Phụ lục B: bảng đóng góp thành viên |
| Tệp nén | `Nhom3_De2.zip` chứa đúng sáu tệp sản phẩm được yêu cầu ở trên: notebook, báo cáo, slide, hai CSV và requirements |

`src/`, `data/`, `docs/` và `tests/` là cấu trúc làm việc của repository; không thay thế sáu tệp sản phẩm trong gói nộp. Chỉ dùng dữ liệu được cấp. Không suy nhãn test từ dữ liệu ngoài gói.

## I. Sản phẩm nhóm — R1 đến R8 (50 điểm)

| Mã | Điểm | Bằng chứng cần có | Nơi kiểm tra / điều kiện nghiệm thu |
|---|---:|---|---|
| R1.1 | 1 | Đủ sáu tệp sản phẩm, đúng tên/định dạng; hai CSV đúng schema, 1.535 dòng, ID test khớp, không trùng/thiếu. | Sáu tệp trong ZIP; kiểm tra tự động CSV và danh sách tệp. |
| R1.2 | 2 | Bản notebook đã chạy từ đầu đến cuối; thời gian chạy; CSV sinh ra từ lần chạy đó khớp tệp nộp. | Lưu notebook có output; ghi thời gian và kết quả kiểm tra tái lập trong checklist nghiệm thu. |
| R1.3 | 1 | Seed cố định; `requirements.txt` khớp thư viện thực dùng; notebook đánh số Phần 1–6; Markdown giải thích giữa các phần mã. | Notebook, `requirements.txt`, log Restart & Run All. |
| R2.1 | 2 | Thống kê có số liệu và biểu đồ: kích thước giỏ, tần suất món/đuôi dài, top 20; so sánh train/test và biến mục tiêu theo đúng thông tin nhãn được phép biết. | Notebook Phần 1 và báo cáo có bảng/hình, trục và đơn vị rõ. Test không có nhãn; không được tự tạo nhãn test. |
| R2.2 | 3 | Đo và minh chứng vấn đề chất lượng dữ liệu theo đề: giỏ một món, giỏ rất lớn, tên gần nghĩa/không thống nhất, giỏ trùng và vấn đề phát hiện thêm. | Notebook Phần 1 và báo cáo: từng vấn đề có cách đếm, số đo và ví dụ phù hợp. |
| R2.3 | 2 | Mỗi quyết định tiền xử lý có lý do và phương án thay thế/so sánh; mọi phép học tham số được fit trong train fold; test không dùng để chọn mô hình/ngưỡng. | Notebook, báo cáo phương pháp và quy trình validation. |
| R3.1 | 2 | Giao dịch/ma trận nhị phân đúng; ít nhất hai cách biểu diễn hoặc cách rời rạc hóa được so sánh, hoặc giải thích có căn cứ vì sao rời rạc hóa không áp dụng cho dữ liệu chỉ gồm món có/không. | Notebook Phần 1–2; ghi rõ biểu diễn, số thuộc tính và quyết định kỹ thuật. Không tạo biến liên tục giả để rời rạc hóa. |
| R3.2 | 2 | Apriori và FP-Growth chạy cùng dữ liệu với ít nhất 5 ngưỡng support trong khoảng 0,2%–2%; bảng số itemset theo kích thước, số luật, thời gian; đối chiếu kết quả và giải thích ngưỡng chọn. | Notebook Phần 2 và bảng tương ứng trong báo cáo. |
| R3.3 | 2 | Chọn/diễn giải luật bằng support, confidence, lift và thêm leverage/conviction; giảm luật dư thừa; giải thích luật phổ biến, whole milk và tương quan không phải nhân quả. | Notebook/báo cáo Phần 2: luật có hướng đúng, số liệu và diễn giải ứng dụng. |
| R3.4 | 2 | Kiểm tra ổn định luật theo 5 lần chia ngẫu nhiên train thành hai nửa; trong 30 luật lift cao của nửa đầu, đếm luật còn lift > 1,2 ở nửa sau; báo cáo số liệu và nhận xét. | Notebook/báo cáo Phần 2; ghi seed/cách chia, tiêu chí và kết quả từng lần. |
| R4.1 | 5 | Dự đoán xác suất trên toàn bộ test bằng mô hình đã chọn; `predictions.csv` đúng định dạng. Chỉ số tự động của Đề 2: AUC (tối đa 2,5 điểm; mức nền 0,50, tham chiếu 0,68) và AP (tối đa 2,5; mức nền là prevalence test, tham chiếu 0,41). | CSV nộp và đánh giá kín của giảng viên. Không đọc/khôi phục nhãn test. |
| R4.2 | 3 | Validation mô phỏng test: chỉ dùng giỏ gốc có ít nhất 2 món, bỏ `other vegetables` khỏi đặc trưng, giữ nhãn; chọn ngưỡng/tinh chỉnh chỉ trên validation/CV, không trên test. | Notebook Phần 3: mô tả cách tạo fold, pipeline và ranh giới train/validation/test. |
| R4.3 | 2 | Ít nhất 3 mô hình thuộc các họ phù hợp; so sánh cùng split/điều kiện và công bố siêu tham số đã thử. Thử ít nhất 3 nhóm đặc trưng theo đề. | Notebook/báo cáo Phần 3: bảng metric và cấu hình. |
| R4.4 | 2 | Báo cáo AUC, AP, precision trong top 10%, calibration; confusion matrix/lớp yếu; món có tác động dương/âm; đối chiếu luật Phần 2 và quyết định ngưỡng sử dụng. | Notebook/báo cáo Phần 3; kết luận lợi ích, chi phí/rủi ro cho cửa hàng. |
| R5.1 | 2 | Biểu diễn giỏ phù hợp dữ liệu thưa/lệch; chạy ít nhất 2 thuật toán phân cụm khác họ; nêu khoảng cách và lý do. | Notebook/báo cáo Phần 4: mô tả tiền xử lý, thuật toán, tham số. |
| R5.2 | 2 | Chọn số cụm bằng ít nhất 2 tiêu chí và kiểm tra ổn định bằng bootstrap/nhiều seed, báo ARI hoặc số đo ổn định tương đương. | Notebook Phần 4: bảng/đồ thị theo k và kết quả ổn định. |
| R5.3 | 2 | Mô tả từng cụm bằng số liệu so với tổng thể; đặt tên cụm dựa trên lift món; đề xuất hành động cụ thể cho mỗi cụm. | Notebook/báo cáo Phần 4: kích thước, đặc trưng, lift và hành động. |
| R5.4 | 1 | Sau khi hoàn tất phân cụm, đối chiếu cụm với biến mục tiêu/chỉ số có thể kiểm tra trên train; giải thích điểm không khớp. Nhãn không được dùng để dựng cụm. | Notebook Phần 4; ghi rõ đối chiếu là hậu kiểm, không phải đầu vào phân cụm. |
| R6.1 | 0 — không áp dụng | Rubric ghi rõ mục này chỉ dành cho Đề 1. | Không cần tạo bằng chứng anomaly AUC có nhãn cho Đề 2; nhãn anomaly test không được cấp. |
| R6.2 | 2 | Ít nhất 2 phương pháp phát hiện bất thường khác nguyên lý; nêu tham số, lý do và cách đưa về score cùng chiều (cao = bất thường). | Notebook/báo cáo Phần 5. |
| R6.3 | 2 | Rà soát thủ công 20 giỏ điểm cao nhất của từng phương pháp; phân loại có căn cứ và đếm mức trùng giữa phương pháp. | Notebook/báo cáo Phần 5: danh sách, item trong giỏ, lý do đánh giá. |
| R6.4 | 1 | Thử giữ/bỏ các giỏ bất thường khỏi train; đo ảnh hưởng lên Phần 2 và Phần 3; kết luận dựa trên số liệu. | Notebook/báo cáo Phần 5: so sánh cùng điều kiện, tránh dùng test để quyết định. |
| R7.1 | 2 | Nêu ít nhất 3 giới hạn cụ thể của dữ liệu và hậu quả từng giới hạn đối với kết luận bài. | Báo cáo Phần 6; notebook có tóm tắt phù hợp. |
| R7.2 | 2 | Kịch bản rủi ro riêng tư/đạo đức/pháp lý gắn với một kết quả thực tế của nhóm; biện pháp giảm thiểu khả thi và điều kiện trước triển khai. | Báo cáo Phần 6; không chỉ nêu khẩu hiệu chung. |
| R7.3 | 1 | Chọn đúng một xu hướng chương 13, chỉ ra điểm yếu cụ thể của hệ thống hiện tại mà xu hướng xử lý và dữ liệu cần bổ sung. | Báo cáo Phần 6; notebook ghi tóm tắt. |
| R8.1 | 1 | Báo cáo theo mẫu; bảng/hình có tiêu đề, chú thích, nguồn; số liệu nhất quán với notebook; phần chính không quá 12 trang. | `baocao.docx` và đối chiếu chéo với notebook. |
| R8.2 | 1 | Văn phong rõ; trích dẫn giáo trình/tài liệu đã đọc; không vượt giới hạn trang; kết luận có số liệu kiểm chứng. | `baocao.docx`, danh mục tài liệu tham khảo. |

## II. Trình bày — S1 đến S3 (10 điểm)

| Mã | Điểm | Bằng chứng cần có | Nơi kiểm tra / điều kiện nghiệm thu |
|---|---:|---|---|
| S1 | 4 | Slide có đủ sáu phần và số liệu chính; tối đa 15 slide; thiết kế dễ đọc; có hạn chế và kết luận. | `slide.pptx`; kiểm tra số slide, nội dung, khả năng đọc. |
| S2 | 4 | Bài thuyết trình mạch lạc trong 25 ± 3 phút; cả nhóm trả lời được câu hỏi chung. | Run-through có bấm giờ; agenda/kịch bản và biên bản luyện hỏi nếu nhóm dùng. |
| S3 | 2 | Chạy trực tiếp một bước notebook, đổi một tham số và giải thích kết quả thay đổi. | Demo đã thử trước; thành viên trình bày hiểu mã và hệ quả tham số. |

## III. Vấn đáp cá nhân — V1 đến V5 (30 điểm)

Mỗi câu 6 điểm. Mỗi thành viên cần tự ôn mọi phần vì giảng viên có thể hỏi phần do người khác phụ trách.

| Mã | Trọng tâm | Bằng chứng hiểu biết cần chuẩn bị |
|---|---|---|
| V1 | Khái niệm gắn với bài | Giải thích support, confidence, lift khác nhau thế nào; vận dụng lên một luật thật của nhóm. |
| V2 | Giải thích lựa chọn | Nêu lý do chọn ngưỡng support, số cụm, mô hình hoặc tham số; so sánh phương án thay thế bằng kết quả. |
| V3 | Đọc mã | Giải thích được một cell/hàm notebook: đầu vào, đầu ra, vị trí trong quy trình và lý do. |
| V4 | Biến đổi tình huống | Dự đoán và giải thích hướng thay đổi khi đổi tham số/bỏ bước; sẵn sàng chạy thử. |
| V5 | Phản biện | Nêu điểm yếu nhất trong kết luận, tác động của giới hạn dữ liệu và một cách kiểm chứng tiếp theo. |

Mức tối đa đòi hỏi trả lời đúng, giải thích lý do và nêu giới hạn/trường hợp ngoại lệ; chỉ đọc lại kết quả là chưa đủ.

## IV. Quá trình — Q1 đến Q3 (10 điểm)

| Mã | Điểm | Bằng chứng cần có | Nơi kiểm tra / điều kiện nghiệm thu |
|---|---:|---|---|
| Q1 | 3 | Checkpoint 1: tài liệu một trang, đúng hạn; ghi nhận đã đọc dữ liệu, có số liệu ban đầu và dự kiến kỹ thuật cho từng phần. | Tệp checkpoint và bằng chứng nộp đúng hạn. Ngày hạn cần xác nhận với giảng viên nếu chưa được công bố. |
| Q2 | 4 | Checkpoint 2: notebook nháp Phần 1–3 chạy được; tham dự buổi gặp 15 phút; bản cuối thể hiện đã sửa theo góp ý. | Notebook nháp, bằng chứng buổi gặp/góp ý và các thay đổi đã thực hiện. Ngày hạn cần xác nhận nếu chưa rõ. |
| Q3 | 3 | Phụ lục A trung thực, đầy đủ: công cụ AI, câu hỏi, kết quả, cách nhóm kiểm chứng/sửa; có ít nhất một ví dụ AI sai được nhóm phát hiện và sửa. | Nhật ký dùng AI trong báo cáo; đối chiếu lịch sử làm việc và khả năng giải thích của thành viên. |

## Quy tắc tính điểm và các cổng nghiệm thu

- Điểm cá nhân theo rubric: `(I + II + IV) × H + III`, sau đó chia 10 để quy về thang 10. Hệ số đóng góp `H` tính theo tỷ lệ đóng góp ở Phụ lục B; phải cộng đúng 100%.
- Nếu vấn đáp dưới 9/30, điểm cá nhân không vượt 50/100. Nếu dưới 12/30, H tối đa 0,85; giảng viên có thể hạ H nếu không hiểu phần khai đã đóng góp.
- Đề 2 có chỉ số tự động ở R4.1: AUC và Average Precision của `prob` trên nhãn test giữ kín. Sai schema/ID/số dòng có thể làm mất điểm phần tự động.
- Không tạo nhãn test, không tải dữ liệu ngoài gói, không tinh chỉnh trên test. Tập test chỉ dùng cuối cùng để xuất dự đoán và score.
- Trước khi đánh dấu một tiêu chí “Hoàn thành”, reviewer cần xác nhận đúng bằng chứng ở cột tương ứng; hiện trạng checklist WBS cần được nhóm cập nhật riêng sau nghiệm thu.


