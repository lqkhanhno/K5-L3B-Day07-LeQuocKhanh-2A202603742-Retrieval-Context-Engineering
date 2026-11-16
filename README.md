# Day 7 · Agentic RAG

**Bài cá nhân** · **Python 3.11+** · **tiếp nối Day 6**

Bài này xây agent trả lời câu hỏi lịch sử Việt Nam trên kho tài liệu Day 6: agent tự gọi tool truy xuất, trả lời có trích dẫn trong câu và biết từ chối khi thiếu bằng chứng.

## Cấu hình chính

- Retrieval baseline: dense cosine trên Chroma `day7_dense`.
- Retrieval lexical: TF-IDF (`TfidfIndex` của starter).
- Retrieval hybrid: RRF (Reciprocal Rank Fusion, k=60).
- Embedding: local `paraphrase-multilingual-MiniLM-L12-v2` (384 chiều), không cần API cho embedding.
- Agent model: `openai/gpt-oss-120b` qua Groq.
- Prompt/adapter: `starter/src/agent.py` và khung `starter/agentic.py`.

## Chuẩn bị

1. Mở thư mục gốc của repo.
2. Đảm bảo `starter/.env` có key agent hợp lệ.
3. Cài dependencies trong môi trường ảo của `starter`.
4. Trên Windows, luôn bật UTF-8 khi chạy test:

```powershell
$env:PYTHONUTF8 = "1"
```

## Lệnh chạy

### 1) Kiểm chứng benchmark

Từ thư mục gốc:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "starter"
starter\.venv\Scripts\python.exe starter\scripts\verify_benchmark.py --chunks data\chunks.json --questions benchmark\questions.json
```

### 2) Nạp index dense

Từ thư mục `starter`:

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe scripts\build_index.py
```

### 3) Chạy retrieval

Từ thư mục gốc:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "starter"
starter\.venv\Scripts\python.exe -m src.run_retrieval --config dense-top3 tfidf-top3 hybrid-top3 --split dev
starter\.venv\Scripts\python.exe -m src.run_retrieval --config hybrid-top5 hybrid-top5-filter --split test
```

### 4) Chạy agent

Từ thư mục gốc:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "starter"
starter\.venv\Scripts\python.exe -m src.run_agent --config agent-tfidf-top3 --split dev
starter\.venv\Scripts\python.exe -m src.run_agent --config agent-tfidf-top3 --split test
starter\.venv\Scripts\python.exe -m src.run_agent --config agent-dense-top3 --split dev
starter\.venv\Scripts\python.exe -m src.run_agent --config agent-dense-top3 --split test
```

## Chạy test

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "starter"
starter\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

## Kết quả đầu ra

- Retrieval: `results/retrieval/<config>-<split>/summary.json` và `traces.json`
- Agent: `results/agent/<config>-<split>/summary.json`, `agent_summary.json`, `traces.json`, `reviews.json`
- Báo cáo cá nhân: `report/individual/2A202603742_LeQuocKhanh.md`
- Thông tin học viên: `STUDENT.md`

## Ghi chú

- Đây là bài không dùng Docker.
- Bộ test offline đã khóa các lỗi từng gặp trong quá trình làm bài.
- `PYTHONPATH=starter` là cần thiết khi chạy `python -m src.*` từ thư mục gốc trên Windows.
