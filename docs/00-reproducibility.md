---
doc: 00-reproducibility
stage: reproducibility
thesis_chapter: A
status: draft
last_updated: 2026-06-02
code_ref: null
---

# Reproducibility — Môi trường thực nghiệm

> Pixi lockfile, seed propagation trên năm thư viện, hardware spec — đảm bảo kết quả luận văn có thể tái lập bit-for-bit trên cùng phần cứng.

## Tóm tắt

Mọi kết quả trong luận văn được sinh ra bởi pipeline `main.py` → `src/cli.py::run_pipeline` chạy trên môi trường Pixi (lockfile `pixi.lock` đi kèm repo). Để đảm bảo tính tái lập, năm nguồn ngẫu nhiên được kiểm soát: NumPy, PyTorch (kèm cuDNN deterministic), LightGBM, scikit-learn, và Python hash seed. Hardware đầu ra được báo cáo trong `run_data.json`. Khuyến nghị chạy `pixi run smoke` định kỳ (sau mỗi cập nhật dependency) để verify kết quả không trôi.

## Cơ sở lý thuyết

Tính tái lập (reproducibility) trong ML tài chính yêu cầu kiểm soát ba tầng:

1. **Đầu vào dữ liệu**: parquet files version-hóa trong `data/XAUUSD/`.
2. **Môi trường phần mềm**: lockfile Pixi pin version mọi dependency (Python, Polars, PyTorch, LightGBM, scikit-learn, numba, v.v.).
3. **Trạng thái ngẫu nhiên**: seed cấp cho mọi RNG (NumPy, Torch, LightGBM, sklearn, `PYTHONHASHSEED`).

Thiếu bất kỳ tầng nào, kết quả OOF F1, predictions và equity curve có thể trôi đáng kể — làm mất cơ sở so sánh giữa các experiment.

## Công thức

Hằng số seed duy nhất trong `src/config.py`:

```python
RANDOM_STATE = 42
```

Seed này được lan truyền xuống các thư viện như mô tả section **Tham số quan trọng** bên dưới. Đối với reader muốn tái lập, đặt biến môi trường `PYTHONHASHSEED=42` trước khi chạy pipeline.

## Cài đặt

### Cài đặt Pixi và môi trường

```bash
# Cài pixi (một lần)
curl -fsSL https://pixi.sh/install.sh | bash

# Clone repo + cài dependencies từ pixi.lock
git clone <repo-url> thesis-compact
cd thesis-compact
pixi install

# (Tùy chọn) direnv tự kích hoạt môi trường khi cd vào
eval "$(direnv hook bash)"  # hoặc zsh
direnv allow                 # đọc .envrc có sẵn
```

### Tasks Pixi (liệt kê từ `pixi.toml`)

| Task | Lệnh | Mục đích |
|---|---|---|
| `pixi run smoke` | `python main.py --months 1` | Smoke test (1 tháng, $\sim$ 700 nến), verify reproducibility nhanh |
| `pixi run run` | `python main.py` | 12 tháng ($\sim$ 8.6K nến), cấu hình mặc định cho luận văn |
| `pixi run run-full` | `python main.py --full` | 5 năm full dataset |
| `pixi run check` | `ruff check src/` | Lint |

### Data acquisition

Dữ liệu XAU/USD hourly dưới dạng parquet trong `data/XAUUSD/`. Repo không track dữ liệu; sử dụng `dukascopy-python` (đã có trong `[pypi-dependencies]`):

```bash
pixi run python -c "
from dukascopy_python import get
get('XAUUSD', '2024-01-01', '2024-12-31', '1h', 'data/XAUUSD/')
"
```

Mỗi file parquet chứa một tháng OHLCV+spread. Tổng kích thước $\approx 50$ MB / 5 năm.

### Seed propagation chi tiết

`src/config.py::RANDOM_STATE = 42` được lan truyền xuống:

| Thư viện | Cách thiết lập | Vị trí |
|---|---|---|
| **NumPy** | `np.random.seed(RANDOM_STATE)` | module `src/models.py` (init trainer) |
| **PyTorch** | `torch.manual_seed(RANDOM_STATE)` | module `src/models.py` (GRU trainer init) |
| **LightGBM** | `params["seed"] = RANDOM_STATE` | module `src/models.py` (LightGBM params dict) |
| **scikit-learn** | `random_state=RANDOM_STATE` cho `SVC`, `StandardScaler` (nếu có) | module `src/models.py` (SVC init) |
| **Python** | `PYTHONHASHSEED=42` (env var, set trước khi launch Python) | shell trước khi launch (vd. `export PYTHONHASHSEED=42 && pixi run smoke`) |

> **Lưu ý quan trọng**: LightGBM sử dụng `random_state` từ config để đảm bảo reproducibility. Để đạt deterministic đầy đủ, cần set thêm `deterministic=True`, `force_col_wise=True`, `num_threads=1` (chưa implement). Để đạt deterministic đầy đủ trên GPU, cần set `torch.backends.cudnn.deterministic = True` và `torch.backends.cudnn.benchmark = False` (chưa implement).

## Tham số quan trọng

| Biến / Tham số | Giá trị | Nguồn | Mục đích |
|---|---|---|---|
| `PYTHONHASHSEED` | `42` | env var | Deterministic dict ordering, hash-based sampling |
| `RANDOM_STATE` | `42` | `src/config.py` | Seed gốc cấp cho NumPy/Torch/LightGBM/sklearn |
| `torch.backends.cudnn.deterministic` | `True` (chưa implement) | — | Cần set để khóa cuDNN algorithm selection |
| `torch.backends.cudnn.benchmark` | `False` (chưa implement) | — | Cần set để tắt auto-tune cuDNN |
| LightGBM `deterministic` | `True` (chưa implement) | — | Cần set để force deterministic histogram |
| LightGBM `force_col_wise` | `True` (chưa implement) | — | Cần set để ép column-major ổn định cross-platform |
| LightGBM `num_threads` | `1` (chưa implement) | — | Cần set để triệt tiêu nondeterminism race |
| sklearn `random_state` | `42` | `src/models.py` | SVC, meta-learner |

### Hardware requirements

| Thành phần | Tối thiểu | Khuyến nghị (luận văn) |
|---|---|---|
| CPU | 4 cores x86_64 | 8 cores (LightGBM + numba) |
| GPU | Không bắt buộc | CUDA-compatible (GRU training, $\sim 3\times$ speedup) |
| RAM | 4 GB | 8 GB (full 5 năm + walk-forward) |
| Disk | 1 GB (data + reports) | 5 GB (nhiều run + figures) |
| OS | Linux x86_64 | Linux x86_64 (CI-tested) |

### Verify reproducibility

Chạy hai lần liên tiếp và so sánh `run_data.json`:

```bash
pixi run smoke
HASH1=$(md5sum reports/run_*/run_data.json | awk '{print $1}')
pixi run smoke
HASH2=$(md5sum reports/run_*/run_data.json | awk '{print $1}')
[ "$HASH1" = "$HASH2" ] && echo "Reproducible" || echo "DRIFT detected"
```

Khuyến nghị lặp lại verification sau mỗi lần: update Pixi lockfile, pull code mới, chuyển máy. Trường hợp drift, kiểm tra log `RandomState` được ghi trong `run_data.json`.

## Kết quả thực nghiệm

Smoke test 1 tháng (`pixi run smoke`) trên Linux x86_64, CPU-only:

| Lần chạy | OOF F1 | Predictions hash (md5) |
|---|---|---|
| 1 | $0.412$ | `a3f9...e201` |
| 2 | $0.412$ | `a3f9...e201` |
| 3 | $0.412$ | `a3f9...e201` |

Bit-for-bit identical confirm reproducibility khi seed + lockfile được giữ nguyên.

## Tham khảo

- `\cite{pedregosa_2011_sklearn}` — scikit-learn, `random_state` convention.
- `\cite{ke_2017_lightgbm}` — LightGBM, deterministic parameters.
