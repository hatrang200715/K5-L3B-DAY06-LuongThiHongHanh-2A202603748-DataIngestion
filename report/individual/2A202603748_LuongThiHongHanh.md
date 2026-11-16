# Báo cáo cá nhân — Day 6

## Dữ liệu và pipeline

- **Chủ đề và nguồn:** Lịch sử âm nhạc, từ *A Complete History of Music: For Schools, Clubs, and Private Reading* (W. J. Baltzell và các cộng tác viên), Project Gutenberg eBook #54392. Snapshot được tải từ [Project Gutenberg](https://www.gutenberg.org/cache/epub/54392/pg54392-images.html); manifest ghi lần truy cập 2026-11-14 và phiên bản phát hành 2017-03-20 (ấn bản gốc 1908).
- **Quyền và phạm vi:** Manifest ghi tác phẩm không thuộc bản quyền Hoa Kỳ và có thể sao chép/phân phối tại Hoa Kỳ theo Project Gutenberg License cùng điều khoản nhãn hiệu; cần tự kiểm tra luật địa phương nếu dùng ngoài Hoa Kỳ. Đây không phải kết luận pháp lý.
- **Số trang:** 544 trang nội dung, các trang in 17–560. Loại ảnh bìa/trang tiêu đề, mục lục v–viii và chỉ mục 561–565. Quy tắc và kiểm tra mẫu ký tự được ghi trong [manifest](../../starter/data/manifest.yaml). Không có trang rỗng trong 544 trang đã nạp; run metadata ghi 29 cảnh báo kiểm tra độ dài trang, cần xem lại các trang ngắn này nếu mở rộng benchmark.
- **Chuẩn hóa và giới hạn:** Parser đọc HTML theo marker trang in, bỏ marker, giải mã HTML entities, chuẩn hóa khoảng trắng và giữ ranh giới đoạn. Snapshot HTML không đánh dấu riêng phần không đánh số trước trang v hoặc trang ix–xvi. Bộ dữ liệu chỉ có một cuốn sách.
- **Embedding và vector store:** OpenRouter route `openai/text-embedding-3-small`, 1536 chiều; model name được ghi trong [run metadata](../../starter/results/run_metadata.json), nhưng không có mã phiên bản bất biến của model. PostgreSQL 16 với image pgvector `0.8.7-pg16`; lưu bền vững ở volume Docker local. Truy xuất xếp hạng theo cosine distance (`<=>`), top-3; mã store không khai báo ANN index nên truy vấn dùng sắp xếp trực tiếp. Lệnh chạy: `cd starter && python -m src.run`.
- **Metadata:** Mỗi chunk giữ `document_id`, `source`, `page`; trường lọc `section` được gán từ trang in: `introduction` (17–23), `lesson_1` (24–34), `lesson_2` (35–45), `lesson_3` (46), `later_content` (47–560). Ranh giới lấy từ [benchmark schema](../../starter/benchmark/questions.json), giúp giới hạn câu hỏi theo bài học. Tổng chunk: A 2.301; B/C 2.192.
- **Chi phí và thời gian:** Không được ghi trong kết quả hiện có; không suy đoán hoặc báo là 0. Thời điểm tạo run metadata là 2026-11-14 16:57:09 UTC.

## Cấu hình và benchmark

- **A:** fixed-size 500 ký tự, overlap 50, không filter.
- **B:** gom theo đoạn, tối đa 500 ký tự, không filter.
- **C:** giống B, chỉ thêm filter `section` khi câu hỏi có filter. B–C chỉ khác việc lọc metadata.
- Cả ba dùng chung corpus, embedding model, store, metric và benchmark; query embedding được tái sử dụng giữa các chiến lược. Cấu hình thực chạy ở [run metadata](../../starter/results/run_metadata.json).
- **Bộ câu hỏi:** 20 câu trong [questions.json](../../starter/benchmark/questions.json), gồm 15 dev / 5 test. Có 4 câu hỏi dùng filter (2 dev, 2 test). Gold gồm nguồn, trang và `evidence_quote`; nhãn truy xuất tự động đòi quote đã chuẩn hóa xuất hiện trong chunk ở đúng nguồn/trang. Tuy nhiên, run metadata xác nhận test đã được xem ở các lượt chạy trước: đây là **exploratory rerun**, không phải blind test độc lập.
- **Tiêu chí chọn:** Không có bằng chứng lưu lại rằng tiêu chí đã được chốt trước khi xem test; do test đã bị xem nên không thể tuyên bố kết quả test là xác nhận độc lập. Để kết luận thận trọng, ưu tiên Evidence Hit@3 trên dev, sau đó Hit@1, khả năng filter đúng và số chunk; dùng test chỉ như kết quả mô tả.

## Kết quả

Hit@1/Evidence Hit@3 yêu cầu chunk chứa đúng evidence quote đã kiểm chứng, không chỉ cùng trang.

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian | Chi phí |
|---|---|---:|---:|---:|---|---|
| A | Dev | 11/15 | 13/15 | 2.301 | Không ghi | Không ghi |
| A | Test* | 5/5 | 5/5 | 2.301 | Không ghi | Không ghi |
| B | Dev | 11/15 | 14/15 | 2.192 | Không ghi | Không ghi |
| B | Test* | 5/5 | 5/5 | 2.192 | Không ghi | Không ghi |
| C | Dev | 11/15 | 14/15 | 2.192 | Không ghi | Không ghi |
| C | Test* | 5/5 | 5/5 | 2.192 | Không ghi | Không ghi |

| Strategy | Coherence trung bình /2 (n) | Metadata đủ và đúng (n/N) | Filter tuân thủ | Grounding thủ công 0/1/2; trung bình /2 |
|---|---|---:|---|---|
| A | Chưa chấm (0/10 nhãn) | 15/15 top-3 bản ghi test | N/A — thiết kế không filter | Chưa chấm (0/5); trung bình N/A |
| B | Chưa chấm (0/10 nhãn) | 15/15 top-3 bản ghi test | N/A — thiết kế không filter | Chưa chấm (0/5); trung bình N/A |
| C | Chưa chấm (0/10 nhãn) | 15/15 top-3 bản ghi test | 2/2 câu test có filter; 6/6 chunk top-3 khớp filter | Chưa chấm (0/5); trung bình N/A |

Metadata được kiểm tra trên 5 câu test × top-3 mỗi chiến lược: source/document ID, page và section đều hiện diện/khớp quy tắc gán trang trong trace. Đây là kiểm tra dữ liệu có cấu trúc, không thay cho audit hữu ích metadata bằng người chấm. Các nhãn coherence và grounding trong [manual_review.json](../../starter/results/manual_review.json) vẫn để trống; tự động Evidence Hit@3 không được dùng thay cho grounding thủ công.

Trace đầy đủ: [traces.json](../../starter/results/traces.json); tổng hợp: [summary.json](../../starter/results/summary.json). Mẫu coherence được chọn theo 10 trang in đầu tiên có trong corpus (17–26), lấy chunk đầu tiên trên mỗi trang cho mỗi chiến lược; mẫu grounding gồm cả 5 câu test cho mỗi chiến lược. Chưa có nhãn thủ công hoặc lý do chấm trong artifact.

*\*Test đã được xem ở lần chạy trước; các số này không phải kết quả blind.*

## Phân tích và kết luận

- **A–B:** Trên dev, Hit@1 giữ nguyên (11/15), Evidence Hit@3 tăng từ 13/15 lên 14/15; B dùng ít hơn 109 chunk (khoảng 4,7%). Ví dụ **q04**: gold ở trang 20, câu trả lời liệt kê “Form, Expression, Melody, Rhythm, Harmony and Instrumental Color.” A không đưa trang 20 vào top-3 (các trang trả về gồm 551, 530, 138); B đưa trang 20 vào top-3 và khớp evidence. Kết quả gợi ý chunk theo đoạn giữ được nội dung liên quan tốt hơn ở trường hợp này.
- **B–C:** Điểm tự động và số chunk giống nhau; khác biệt có ích là C thực thi filter. Hai câu test có filter (q16, q20) đều trả về top-3 đúng `section` (2/2 câu, 6/6 chunk). Đây chỉ là hai câu test đã từng được xem, chưa đủ để kết luận filter cải thiện độ chính xác tổng thể.
- **Ví dụ lỗi còn lại — q07:** Cả A/B/C không đạt Evidence Hit@3. Gold quote về giả thuyết Darwin nằm ở trang 22; các top-3 có trang 22 nhưng chunk không chứa trọn quote (A/B/C đều bị tính miss theo tiêu chí evidence chính xác). Đây là lỗi truy xuất theo ranh giới chunk/evidence, không phải bằng chứng rằng nguồn không có câu trả lời.
- **Lựa chọn tạm thời:** Chọn C khi câu hỏi có `section` do hỗ trợ lọc đúng mà không làm tăng chunk so với B; khi không có filter, B và C có kết quả như nhau trong run này. A có nhiều chunk hơn và thấp hơn B/C một hit Evidence Hit@3 trên dev. Lựa chọn chỉ có giá trị trên corpus và benchmark hiện tại.
- **Hạn chế:** Một nguồn duy nhất, 20 câu và chỉ 5 câu test; test đã bị xem trước ở các lượt chạy trước; tiêu chí chọn trước test không được ghi lại. Nhãn coherence/grounding chưa hoàn thành. Thời gian và chi phí API không được log. Embedding model là tên model theo route, chưa pin phiên bản bất biến; section gán theo dải trang thủ công; exact quote matching có thể bỏ sót bằng chứng tương đương hoặc evidence bị chia qua ranh giới chunk.
- **Nếu làm lại/mở rộng:** Tạo test mới chưa xem, chốt trước tiêu chí chọn, tăng số câu theo nhiều bài/điều kiện, audit độc lập đủ 10 mẫu coherence và 5 câu grounding mỗi chiến lược, ghi nhãn cùng lý do. Đo riêng thời gian ingest/query và token/chi phí provider; kiểm tra 29 cảnh báo độ dài trang; bổ sung đánh giá evidence đa chunk cho trường hợp như q07.
