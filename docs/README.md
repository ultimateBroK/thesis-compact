---
thesis_title: "Dự báo tín hiệu giao dịch XAU/USD bằng stacking ensemble với purged-embargo cross-validation"
author: Nguyen Duc Hieu
supervisor: [Chưa cung cấp]
university: [Chưa cung cấp]
department: [Chưa cung cấp]
last_updated: 2026-06-03
status: draft
doc: docs/README.md
stage: overview
thesis_chapter: null
code_ref: null
---

# Tài liệu luận văn

> Bộ tài liệu phục vụ luận văn về pipeline stacking ensemble (GRU + LightGBM + SVC) dự báo tín hiệu XAU/USD với purged-embargo cross-validation, triple-barrier labeling và meta-labeling.

## Tổng quan luận văn

Luận văn đề xuất một pipeline học máy hoàn chỉnh để dự báo tín hiệu giao dịch vàng (XAU/USD) trên khung hourly. Đóng góp chính là việc kết hợp ba mô hình không đồng nhất (GRU nơ-ron truy hồi, LightGBM gradient boosting, SVC support vector) trong kiến trúc *stacking ensemble* cùng với các kỹ thuật chuyên biệt cho tài chính: fractional differencing để bảo toàn memory, triple-barrier labeling cho gán nhãn không đối xứng, meta-labeling cho position sizing, và purged-embargo cross-validation để phòng tránh rò rỉ thông tin.

### Pipeline

```mermaid
flowchart LR
    A[Tick Parquet] --> B[OHLC 1h]
    B --> C[21 Features]
    C --> D[Triple-Barrier Labels]
    D --> E[Train/Test Split]
    E --> F[GRU + LightGBM + SVC]
    F --> G[Stacking Ensemble]
    G --> H[Backtest + Reports]
```

### Cấu trúc năm chương

| Chương | Nội dung | Doc liên quan |
|---|---|---|
| **Chương 1** — Giới thiệu | Bối cảnh, mục tiêu, đóng góp | `README.md` (file này) |
| **Chương 2** — Cơ sở lý thuyết | Fractional diff, triple-barrier, meta-labeling, purged CV, stacking | `10-`…`14-methodology-*.md` |
| **Chương 3** — Phương pháp đề xuất | Pipeline chi tiết per-module | `01-`…`09-*.md` |
| **Chương 4** — Thực nghiệm | EDA, thiết kế thí nghiệm, đánh giá, kết quả | `20-`…`22-*.md`, `30-exploratory-analysis.md` |
| **Chương 5** — Kết luận | Tổng kết, hạn chế, hướng phát triển | Viết riêng trong thesis LaTeX |

## Đối tượng độc giả

- **Hội đồng chấm luận văn**: tham gia chương 2–4, danh mục thuật ngữ (Phụ lục B), môi trường thực nghiệm (Phụ lục A).
- **Nhà nghiên cứu tiếp theo**: tái lập kết quả, mở rộng sang các tài sản khác (forex, equity index). Đọc theo thứ tự `00-reproducibility.md` → `01-`…`09-` (pipeline) → `10-`…`14-` (lý thuyết) → `20-`…`22-` (thí nghiệm).
- **Practitioner ML tài chính**: tham khảo quick-start ở `README.md` gốc + `09-cli.md`.

## Ánh xạ tài liệu — chương luận văn

| File | Chương | Trạng thái |
|---|---|---|
| `00-reproducibility.md` | Phụ lục A — Môi trường thực nghiệm | draft |
| `01-data-pipeline.md` | Chương 3 | stub |
| `02-features.md` | Chương 3 | draft |
| `03-labeling-triple-barrier.md` | Chương 3 | stub |
| `04-validation-purged-embargo.md` | Chương 3 | stub |
| `05-models-stacking.md` | Chương 3 | stub |
| `06-backtest.md` | Chương 3 | stub |
| `07-reporting.md` | Chương 3 | stub |
| `08-config.md` | Chương 3 | draft |
| `09-cli.md` | Chương 3 | stub |
| `10-methodology-fracdiff.md` | Chương 2 | draft |
| `11-methodology-triple-barrier.md` | Chương 2 | stub |
| `12-methodology-meta-labeling.md` | Chương 2 | stub |
| `13-methodology-purged-cv.md` | Chương 2 | draft |
| `14-methodology-stacking.md` | Chương 2 | stub |
| `20-experiments.md` | Chương 4 | stub |
| `21-results-convention.md` | Chương 4 | stub |
| `22-evaluation-metrics.md` | Chương 4 | stub |
| `30-exploratory-analysis.md` | Chương 4 | stub |
| `90-glossary.md` | Phụ lục B — Danh mục thuật ngữ | stub |
| `references.bib` | Nguồn cite duy nhất (IEEE numeric) | seed 13 entries |

## Cấu trúc thư mục

- `00-reproducibility.md` — Môi trường Pixi, seed control trên 5 thư viện, data acquisition, hardware spec phục vụ tái lập kết quả.
- `01-data-pipeline.md` — Parquet → OHLC Polars streaming, timeframe 1h, cấu trúc dữ liệu XAU/USD.
- `02-features.md` — 21 đặc trưng đầu vào (19 dẫn xuất + 2 raw passthrough) cho stacking ensemble: fractional differencing, chỉ báo kỹ thuật, OBV, calendar.
- `03-labeling-triple-barrier.md` — Swing H/L + ATR fallback + auto-tune barriers + meta-labeling cho position sizing.
- `04-validation-purged-embargo.md` — PurgedEmbargoTimeSeriesSplit, phòng tránh leakage trong time series tài chính.
- `05-models-stacking.md` — GRU + LightGBM + SVC + meta-learner + smart filtering — core contribution của luận văn.
- `06-backtest.md` — Barrier-based equity sim, leverage, position sizing trên chuỗi trade giả lập.
- `07-reporting.md` — `reports/run_*/` artifacts, figures, JSON schema cho mỗi lần chạy pipeline.
- `08-config.md` — Toàn bộ hyperparams trong `src/config.py` kèm rationale cho từng tham số.
- `09-cli.md` — CLI orchestration qua `main.py`, flow subcommand, exit codes, logging.
- `10-methodology-fracdiff.md` — Fractional differencing ($0 < d < 1$): toán lý thuyết và lý do chọn $d = 0.4$.
- `11-methodology-triple-barrier.md` — Lý thuyết triple-barrier của López de Prado cho gán nhãn chuỗi thời gian tài chính.
- `12-methodology-meta-labeling.md` — Meta-labeling nhằm cải thiện position sizing và giảm false positives.
- `13-methodology-purged-cv.md` — Purged k-fold kèm embargo cho financial data, phòng tránh information leakage.
- `14-methodology-stacking.md` — Lý thuyết stacking ensemble và out-of-fold predictions cho kết hợp mô hình không đồng nhất.
- `20-experiments.md` — Hyperparameter grid, seed sweep, ablation design phục vụ so sánh mô hình.
- `21-results-convention.md` — Cách đọc `reports/run_*`, bảng kết quả chuẩn, ablation matrix cho luận văn.
- `22-evaluation-metrics.md` — F1, Sharpe, max-drawdown, win rate, PnL — định nghĩa và công thức tính.
- `30-exploratory-analysis.md` — Convert từ `viz.ipynb` sang markdown phục vụ trích dẫn trong luận văn.
- `90-glossary.md` — Thuật ngữ VI/EN: TP, SL, ATR, OOF, embargo, stacking, fractional diff, meta-labeling.
- `references.bib` — BibTeX (IEEE numeric), single source of truth, 13 entries.

## Quy ước viết

- **Ngôn ngữ chính**: tiếng Việt; thuật ngữ kỹ thuật giữ tiếng Anh trong ngoặc (vd: "thanh chắn ba lớp (triple-barrier)").
- **Tên file**: `kebab-case` với tiền tố số `NN-` đảm bảo thứ tự sort tự nhiên khi `ls docs/`.
- **Không dùng suffix ngôn ngữ** (`*.vi.md`); nếu cần bản EN thì tạo `docs-en/` riêng.
- **Citation style**: IEEE numeric, cite từ `docs/references.bib` duy nhất bằng cú pháp `\cite{key}`.
- **Frontmatter**: mỗi doc có YAML bắt buộc với 6 trường: `doc`, `stage`, `thesis_chapter`, `status`, `last_updated`, `code_ref`. Docs overview (như file này) bổ sung thêm metadata (thesis_title, author, supervisor, university, department).
- **Stub template**: 7 section cố định:

  1. **Tóm tắt** — Ý chính doc (1-2 câu)
  2. **Cơ sở lý thuyết** — Background lý thuyết liên quan (methodology docs bỏ section này)
  3. **Công thức** — Công thức/toán ký hiệu cần thiết
  4. **Cài đặt** — Implementation details, code refs (methodology docs bỏ section này)
  5. **Tham số quan trọng** — Hyperparams + rationale
  6. **Kết quả thực nghiệm** — Output, metrics, observations
  7. **Tham khảo** — `\cite{}` keys đã dùng trong doc
- **Methodology docs** (10–14) bỏ section `Cài đặt` vì lý thuyết thuần.
- **Module docs** (01–07) có thể bỏ `Cơ sở lý thuyết` nếu trùng methodology doc tương ứng.
- **Math**: inline `$...$`, display `$$...$$`, LaTeX syntax chuẩn.
- **Code refs**: cú pháp `src/path/file.py::function_name`.

## Cách đóng góp / Cập nhật

### Quy trình cập nhật một doc

1. Sửa nội dung markdown trực tiếp (không cần script regeneration — stub chỉ chạy một lần bởi `tools/gen_doc_stubs.py`).
2. Cập nhật `last_updated: YYYY-MM-DD` ở frontmatter với ngày sửa.
3. Nếu code được tài liệu hóa thay đổi, cập nhật `code_ref` với commit SHA mới (lấy từ `git log -1 --format=%H src/path/file.py`).
4. Kiểm tra không còn placeholder `[Chưa viết]` — nếu còn, đánh dấu `status: stub`, ngược lại `status: draft`.
5. Chạy `pixi run check` để lint (cho code block Python trong doc, ruff không check markdown — nhưng đảm bảo code trong doc hợp lệ).

### Khi nào bump `last_updated`

- Mỗi lần sửa nội dung substantitive (thêm section, thay đổi giá trị tham số, bổ sung kết quả).
- Không bump cho typo nhỏ hoặc format whitespace.

### Khi nào update `code_ref`

- Sau mỗi commit thay đổi `src/` được tài liệu hóa trong doc.
- Đặt `code_ref: null` khi doc chưa pin đến commit cụ thể (init phase).

### Trạng thái doc

| `status` | Ý nghĩa |
|---|---|
| `stub` | Chỉ có frontmatter + section headers, nội dung placeholder `[Chưa viết]` |
| `draft` | Nội dung đầy đủ, chưa review |
| `review` | Đã self-review, chờ supervisor |
| `final` | Đã góp ý + chỉnh sửa, sẵn sàng paste vào thesis LaTeX |

## Tham khảo

BibTeX entries được lưu tại `docs/references.bib`. Mỗi doc cite bằng cú pháp `\cite{key}` — listing tham khảo cuối mỗi doc chỉ liệt kê các key đã dùng trong doc đó.
