# DBM301 – Assignment 2 – Đề 2: Phân tích giỏ hàng Groceries

Repository làm bài của **Nhóm 3** cho học phần Khai phá dữ liệu. Nguồn yêu cầu là `De_Assignment2_De2.docx` do giảng viên cung cấp; [bảng đối chiếu rubric](docs/rubric_evidence_matrix.md) dùng để rà soát minh chứng trước khi nộp.

## Bài toán và dữ liệu

Bài làm nghiên cứu các món thường được mua cùng nhau, dự đoán xác suất giỏ hàng ban đầu có **`other vegetables`**, phân cụm kiểu mua sắm và phát hiện giỏ bất thường. Dữ liệu chỉ chứa danh sách món trong mỗi giỏ; không có giá, số lượng, thời điểm hay thông tin khách hàng. Vì vậy, luật kết hợp và cụm chỉ hỗ trợ giả thuyết kinh doanh, không chứng minh quan hệ nhân quả.

Chỉ dùng ba tệp được cấp trong `data/raw/`:

| Tệp | Nội dung |
| --- | --- |
| `groceries_train.csv` | 8.300 giỏ, hai cột `basket_id,items`; có giỏ một món. |
| `groceries_test.csv` | 1.535 giỏ, hai cột `basket_id,items`; `other vegetables` đã bị xóa khỏi danh sách món. Giỏ test bắt nguồn từ giỏ có ít nhất hai món. |
| `data_dictionary.txt` | Mô tả trường dữ liệu và quy ước của đề. |

`items` là danh sách tên món phân cách bằng dấu chấm phẩy (`;`). Nhãn test được giữ kín: **không suy ra nhãn âm từ việc không thấy `other vegetables` trong test**, không tải bộ Groceries gốc hoặc dữ liệu bên ngoài để khôi phục nhãn.

## Nội dung phải hoàn thành

| Phần trong đề | Nội dung chính |
| --- | --- |
| 1. Tìm hiểu và chuẩn bị | Tách giao dịch, EDA và kiểm tra chất lượng; so tỷ lệ mục tiêu giữa toàn train và nhóm giỏ từ hai món; ánh xạ 169 món vào ít nhất 15 nhóm hàng. |
| 2. Mẫu thường xuyên và luật kết hợp | So Apriori với FP-Growth ở ít nhất năm mức support từ 0,2% đến 2%; đánh giá luật, giảm dư thừa, luật mức nhóm, kiểm tra ổn định qua năm lần chia và đề xuất năm ứng dụng có điều kiện. |
| 3. Phân lớp | Validation mô phỏng cách tạo test; ít nhất ba mô hình và ba bộ đặc trưng; so với bộ phân lớp dựa trên luật; báo cáo AUC, AP, precision@10%, hiệu chuẩn và sinh xác suất test. |
| 4. Phân cụm | Ít nhất hai thuật toán; chọn số cụm bằng hai tiêu chí và kiểm tra độ ổn định; mô tả cụm bằng lift, kiểm tra ảnh hưởng kích thước giỏ và đo tác động khi thêm nhãn cụm vào Phần 3. |
| 5. Bất thường | Ít nhất hai cách chấm theo các nguyên lý khác nhau; đọc 20 giỏ đứng đầu mỗi cách, đo độ trùng và thử ảnh hưởng của việc loại giỏ bất thường lên Phần 2–3. |
| 6. Phản biện, rủi ro và xu hướng | Nêu ít nhất ba giới hạn dữ liệu và hệ quả; kịch bản rủi ro riêng tư gắn với kết quả; chọn một xu hướng chương 13 và dữ liệu bổ sung cần có. |

Mỗi phần cần có **mã chạy được trong notebook, số liệu và nhận xét trong báo cáo, nội dung chính trong slide**. Mọi lựa chọn kỹ thuật cần có lý do và phương án đối chiếu. Các bước học tham số hoặc chọn ngưỡng phải nằm trong phần train của từng lượt kiểm định; test chỉ dùng để xuất kết quả cuối.

## Cấu trúc repository

```text
data/raw/                    Dữ liệu giảng viên cấp; không đưa lên Git
data/mappings/               Ánh xạ món sang nhóm và phương án thay thế
notebooks/notebook.ipynb     Notebook phân tích chính
outputs/                    Bảng, hình, predictions.csv, anomaly_scores.csv
src/groceries_group3/       Hàm Python tái sử dụng cho quá trình phát triển
tests/                      Kiểm tra cho mã tái sử dụng
reports/                    Nơi chuẩn bị báo cáo và slide
docs/                       Bảng đối chiếu rubric và tài liệu phối hợp
```

Notebook dùng seed `42` và đọc dữ liệu từ `data/raw/` khi chạy tại gốc repository hoặc từ thư mục `notebooks/`. Các tệp trong `outputs/` là kết quả làm việc, **không thay thế** sáu tệp của gói nộp.

## Thiết lập và chạy

Dự án khai báo Python **3.13** trong `.python-version` và các phụ thuộc trong `pyproject.toml`/`uv.lock`. Cài [uv](https://docs.astral.sh/uv/) rồi thực hiện tại gốc repository:

```powershell
uv sync
uv run python -m unittest discover -s tests
uv run jupyter lab notebooks/notebook.ipynb
```

Đặt ba tệp dữ liệu được cấp vào `data/raw/` trước khi chạy notebook. Trong Jupyter, chọn kernel của `.venv`, chạy **Restart & Run All** và lưu notebook có output. Giới hạn theo đề: chạy từ đầu đến cuối trong **15 phút trên máy 8 GB RAM**. Khi thay đổi thư viện, cập nhật `requirements.txt` với tên và phiên bản **thực sự dùng**, rồi kiểm tra lại trong môi trường sạch.

## Quy cách nộp

Nộp `Nhom3_De2.zip` lên Drive giảng viên cung cấp. Tệp nén phải chứa **đúng sáu tệp** sau ở cấp gốc:

| Tên tệp | Yêu cầu |
| --- | --- |
| `notebook.ipynb` | Một notebook duy nhất, đánh số Phần 1–6, đã chạy và lưu output. |
| `baocao.docx` | Theo `Mau_BaoCao.docx`; phần chính tối đa 12 trang A4. Phụ lục A ghi nhật ký dùng AI, Phụ lục B ghi đóng góp thành viên. |
| `slide.pptx` | Tối đa 15 slide, trình bày trong 25 phút; có kết quả định lượng và giới hạn. |
| `predictions.csv` | Đúng 1.535 dòng, hai cột theo thứ tự `basket_id,prob`; `prob` từ 0 đến 1. |
| `anomaly_scores.csv` | Đúng 1.535 dòng, hai cột theo thứ tự `basket_id,score`; điểm lớn hơn nghĩa là bất thường hơn. |
| `requirements.txt` | Thư viện và phiên bản đã dùng. |

Hai CSV phải có đủ và đúng `basket_id` của test, không trùng hoặc thiếu dòng. Không đưa dữ liệu nguồn, mã phụ trợ hay các bảng trung gian vào tệp nén.

## Trạng thái cần nghiệm thu

Notebook hiện có sáu phần theo đề, đã chạy lại liên tục 36 code cell trong khoảng 52 giây trên môi trường hiện tại. Phần 6 dùng số liệu từ các phần trước để bàn về giới hạn, rủi ro và hệ gợi ý. `outputs/predictions.csv` và `outputs/anomaly_scores.csv` có 1.535 dòng với schema yêu cầu. Đây là **trạng thái làm việc**, chưa phải xác nhận bài nộp hoàn tất: cần hoàn thành `baocao.docx` và `slide.pptx` (thư mục `reports/` hiện chưa có), đối chiếu minh chứng từng mục của đề, kiểm tra lại thời gian chạy trên máy 8 GB RAM và đóng gói đúng sáu tệp.

Việc dùng AI phải được ghi trung thực trong Phụ lục A: công cụ, câu hỏi, kết quả nhận được và cách nhóm kiểm tra hoặc sửa. Mọi thành viên cần giải thích được các phần của bài khi vấn đáp.
