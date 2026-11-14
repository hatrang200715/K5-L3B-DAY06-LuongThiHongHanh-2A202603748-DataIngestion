# Starter · PostgreSQL + pgvector

Bộ khởi động **tùy chọn** cho lab cá nhân. Docker Compose tạo PostgreSQL + pgvector local. Pipeline trong `src/` đọc corpus, tạo embedding, lưu vector và chạy benchmark; `app.py` cung cấp chatbot web để demo retrieval. Vẫn có thể chọn Qdrant, Chroma hoặc vector store tương đương theo [GUIDE.md](../GUIDE.md).

## Cấu trúc

```text
starter/
├── docker-compose.yml     # PostgreSQL 16 + pgvector, chỉ mở cổng local
├── init.sql               # bật extension khi tạo database mới
├── requirements.txt       # dependencies của pipeline và Streamlit
├── .env.example           # cấu hình mẫu; copy thành .env
├── scripts/check_setup.py # kiểm tra kết nối và extension (đã chạy được)
├── src/
│   ├── models.py          # cấu trúc dữ liệu mẫu
│   ├── ingestion.py       # đọc / chuẩn hóa tài liệu
│   ├── chunking.py        # chia chunk
│   ├── embeddings.py      # API embedding adapters
│   ├── store.py           # lưu vector, metadata và filter
│   ├── evaluation.py      # chấm truy xuất, lưu trace
│   ├── chatbot.py         # retrieval + grounded chat completion
│   └── run.py             # kết nối các bước thành pipeline
├── app.py                 # chatbot demo Streamlit, trả lời kèm nguồn trang
├── data/                  # bạn tự chuẩn bị corpus được phép sử dụng
└── results/               # lưu kết quả của bạn
```

## 1. Cài và kiểm tra database

Cần Docker Desktop hoặc Docker Engine + Compose, Python **3.11–3.14**, và mạng để tải image/dependencies lần đầu. Các lệnh sau chạy từ thư mục `starter/`:

```bash
cd starter
python3 -m venv .venv
```

Kích hoạt `.venv`: macOS/Linux dùng `source .venv/bin/activate`; Windows PowerShell dùng `.venv\Scripts\Activate.ps1`. Sau đó:

```bash
python -m pip install -r requirements.txt
# macOS/Linux: cp .env.example .env
# Windows PowerShell: Copy-Item .env.example .env
docker compose -p day6_student_starter up -d --wait
python scripts/check_setup.py
```

Kỳ vọng: hiện tên database, phiên bản PostgreSQL và pgvector, sau đó `Setup OK`. Đây **chỉ là kiểm tra hạ tầng**, không phải kết quả benchmark. Docker dùng cổng `127.0.0.1:55433` và mật khẩu cố định **chỉ dùng local cho lab**; nếu cổng bận, sửa đồng thời port trong `docker-compose.yml` và `DATABASE_URL` trong `.env`. `init.sql` chỉ chạy khi volume mới được tạo; nếu database sẵn có chưa bật extension, xem log Docker và kiểm tra cấu hình thay vì xóa volume chứa dữ liệu.

Dừng container nhưng giữ dữ liệu:

```bash
docker compose -p day6_student_starter stop
```

## 2. Chọn embedding model

Điền key tương ứng trong `.env` nếu dùng OpenRouter, OpenAI hoặc Gemini; **không commit `.env`**. Có thể đặt tên model trần, ví dụ `text-embedding-3-small`, khi chỉ có đúng một provider key; với nhiều key, chỉ rõ `provider:model`, ví dụ `openai:text-embedding-3-small`, `openrouter:openai/text-embedding-3-small` hoặc `gemini:gemini-embedding-001`. Nếu adapter không nhận diện số chiều của model, đặt `EMBEDDING_DIMENSIONS` theo tài liệu provider. Không dùng chung giá trị dimension nếu model trả kích thước vector khác. Ghi provider, model và số chiều thực tế; giữ nguyên model giữa các chiến lược. Adapter API dùng thư viện chuẩn Python, không cần cài SDK riêng.

## 3. Hoàn thiện pipeline

1. Corpus mẫu, manifest và quyền dữ liệu được ghi tại `data/manifest.yaml`. Tải HTML từ Project Gutenberg vào `data/pg54392-images.html` bằng lệnh `curl -L --fail --output data/pg54392-images.html https://www.gutenberg.org/cache/epub/54392/pg54392-images.html`, hoặc đặt đường dẫn khác vào `CORPUS_HTML_PATH`. Theo quy tắc corpus trong [GUIDE.md](../GUIDE.md), pipeline lấy các trang nội dung in 17–560; loại bìa, mục lục v–viii và mục lục cuối sách 561–565. Kiểm tra snapshot không thấy trang trắng hoặc trang chỉ có tài liệu tham khảo trong phạm vi nội dung; trang có cả tài liệu tham khảo lẫn câu hỏi được giữ lại. Bản HTML không có marker riêng cho phần không đánh số trước trang v hoặc các trang ix–xvi.
2. Hoàn thiện từng hàm TODO trong `src/`. Lưu ID tài liệu, nguồn, trang/vị trí và ít nhất một trường filter có ích.
3. Pipeline chạy ba cấu hình trên cùng corpus, model và benchmark: A fixed-size 500 ký tự/overlap 50; B gom theo đoạn đến 500 ký tự; C dùng chính chunk/vector của B và chỉ thêm filter `section` khi câu hỏi có filter. Mỗi query chỉ được embed một lần rồi dùng lại cho cả ba cấu hình. Với mỗi strategy, store xóa và thay toàn bộ chunk trong cùng transaction, nên chạy lại không nhân đôi và dữ liệu/chunk cũ bị loại khi corpus đổi. SHA-256 corpus và benchmark được ghi vào `results/run_metadata.json` để nhận diện snapshot.
4. Tự viết ít nhất 20 câu hỏi kèm evidence, tách dev/test trước khi thử theo [BENCHMARK.md](../BENCHMARK.md). Lưu trace top-3 và báo cáo đủ năm tiêu chí.
5. Sau khi cấu hình `.env`, tải corpus và khởi động PostgreSQL + pgvector, chạy `python -m src.run` từ thư mục `starter/`. Kết quả được lưu trong `results/`: metadata cấu hình, trace top-3, tổng hợp Hit@1/Evidence Hit@3 và các mẫu chấm thủ công chưa có nhãn.

Lưu code, cấu hình, kết quả và lệnh chạy để người chấm tái lập; không commit key, dữ liệu mật hoặc dữ liệu không có quyền chia sẻ.

## 4. Demo chatbot

Sau khi chạy pipeline và đã có vector cho A/B/C trong PostgreSQL:

1. Điền `CHAT_MODEL` trong `.env`, dùng cùng provider API key đã cấu hình cho embedding. Ví dụ OpenRouter: `CHAT_MODEL=openai/gpt-4o-mini`; OpenAI: `CHAT_MODEL=gpt-4o-mini`; Gemini: `CHAT_MODEL=gemini-2.0-flash`. Chat completion là API riêng và có thể phát sinh phí.
2. Từ thư mục `starter/`, chạy `streamlit run app.py` rồi mở URL Streamlit hiển thị trong terminal.
3. Chọn A/B/C, nhập câu hỏi. Bot embedding câu hỏi, truy xuất top-3 từ PostgreSQL, rồi tổng hợp câu trả lời chỉ từ các đoạn đó và hiển thị trang, section, similarity. Chỉ C áp dụng section filter; A/B luôn không lọc như thiết kế benchmark.

Chatbot kiểm tra model embedding đang cấu hình có khớp model dùng để tạo vector hay không. Nếu thiếu dữ liệu, API key hoặc chat model, cần cấu hình/ingest trước; app không dùng dữ liệu hay câu trả lời giả lập. Câu hỏi và các đoạn top-3 sẽ được gửi đến provider chat đã cấu hình; chỉ dùng dữ liệu được phép chia sẻ.
