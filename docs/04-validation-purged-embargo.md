---
doc: 04-validation-purged-embargo
stage: validation
thesis_chapter: 3
status: draft
last_updated: 2026-06-02
code_ref: src/validation.py
---

# Validation — Purged Embargo Time-series Split

> Lớp `PurgedEmbargoTimeSeriesSplit` trong `src/validation.py` (86 dòng): chia $K$-fold contiguous theo thời gian, purge các observation train có label overlap với test, embargo block sau mỗi test fold. Tránh leakage trong OOF cross-validation.

## Tóm tắt

Cross-validation chuẩn (K-fold shuffle) vi phạm giả định IID trong tài chính — autocorrelation và event-overlap gây information leakage từ train sang test \cite{de_prado_2018_afml}. Module `src/validation.py` triển khai hai splitter: (i) `PurgedEmbargoTimeSeriesSplit` cho OOF$k$-fold trong `HybridStackingSignalClassifier.fit` (xem `docs/05-models-stacking.md`); (ii) `walk_forward_split` cho walk-forward evaluation theo năm mở rộng. Cả hai đều giữ tính thứ tự thời gian, không shuffle, và thao tác trực tiếp trên numpy index array để tương thích sklearn API.

## Cơ sở lý thuyết

Lý thuyết đầy đủ ở `docs/13-methodology-purged-cv.md`. Tóm tắt: tại mỗi fold $i$, tập test là block contiguous $\mathcal{T}_i = [s_i, e_i]$ theo thời gian. Tập train gốc là phần bù, nhưng phải loại bỏ:

1. **Purge**: các observation $j$ có label window $[j, \text{event\_end}_j]$ overlap với $\mathcal{T}_i$ — do label dùng future return trong horizon $h=24$, ta phải loại những điểm có "future" nằm trong test.
2. **Embargo**: block $[e_i + 1, e_i + g]$ ngay sau test — bảo vệ chống leakage ngược từ test vào train do các feature rolling có nhìn lùi.

## Công thức

Cho $N$ observation, $K$ fold, $\phi$ = embargo fraction:

$$
n_{\text{test}} = \left\lfloor \frac{N}{K+1} \right\rfloor, \qquad
g = \lceil N \cdot \phi \rceil.
$$

Fold $i$ ($i = 0, \ldots, K-1$):

$$
s_i = (i+1) \cdot n_{\text{test}}, \qquad
e_i = \begin{cases} s_i + n_{\text{test}} - 1, & i < K-1, \\ N - 1, & i = K-1. \end{cases}
$$

Purge mask trên train:

$$
\text{train}_i = \bigl\{j \notin [s_i, e_i] : \neg\bigl(j \leq e_i \;\wedge\; \text{event\_end}_j \geq s_i\bigr)\bigr\} \setminus [e_i + 1, \min(N, e_i + g + 1)].
$$

Điều kiện purge $\neg(\cdot)$ tương đương: giữ $j$ nếu HOẶC $j > e_i$ (sau test, chưa bị embargo chạm) HOẶC $\text{event\_end}_j < s_i$ (label kết thúc trước test bắt đầu).

## Cài đặt

### Cấu trúc lớp

`PurgedEmbargoTimeSeriesSplit` là **plain class** (không kế thừa `BaseCrossValidator` của sklearn) nhưng expose `split(X, event_end)` yield `(train_idx, test_idx)` — tương thích với cách gọi thủ công trong `src/models.py::cross_validate_oof_probabilities`. Lý do không kế thừa: tránh buộc `get_n_splits` contract, dễ pass thêm `event_end` (sklearn BaseCrossValidator chỉ nhận `X, y, groups`).

### Algorithm chi tiết

```
PurgedEmbargoTimeSeriesSplit(n_splits=5, embargo_pct=0.02)
├── __init__(n_splits, embargo_pct)
└── split(X, event_end)   → generator of (train_idx, test_idx)
    ├── indices   = np.arange(len(X))
    ├── event_end_pos = event_end.to_numpy().astype(int)
    ├── embargo   = ceil(len(X) * embargo_pct)
    ├── test_size = max(1, len(X) // (n_splits + 1))
    └── for i in 0..n_splits-1:
         ├── train_end = (i + 1) * test_size
         ├── test_start = train_end
         ├── test_end = test_start + test_size  (hoặc len(X) nếu i == n_splits-1)
         ├── test_idx = indices[test_start:test_end]
         ├── candidate = compute_purged_train_indices(
         │     indices, event_end_pos, test_idx, embargo)
         ├── train_idx = candidate[candidate < test_start]
         └── if len(train_idx) > 0: yield (train_idx, test_idx)
```

Hàm phụ `compute_purged_train_indices(indices, event_end_pos, test_idx, embargo)`:

1. Khởi tạo `train_mask` = all True.
2. Loại test_idx khỏi mask.
3. Loại các $j$ thỏa $j \leq \text{test\_end}$ và $\text{event\_end}_j \geq \text{test\_start}$ (label overlap với test).
4. Loại block $[\text{test\_end}+1, \min(N, \text{test\_end}+\text{embargo}+1)]$ (embargo).
5. Return `indices[train_mask]`.

Edge case:
- **Fold đầu** ($i=0$): train là $[\text{test\_size}, N)$ sau purge — luôn có đủ data vì test_size $\approx N/6$.
- **Fold cuối** ($i=K-1$): `test_end = N` — fold test chiếm phần cuối, không embargo phía sau.
- **Train rỗng**: nếu sau purge train_idx trống thì fold bị skip (`if len(train_idx)` guard).

### Walk-forward split (phụ)

`walk_forward_split(dates, n_windows=3)`: chia theo năm tự nhiên, mỗi window train = mọi năm trước test year (expanding). Yield `(train_idx, test_idx, window_id, train_range, test_range)`. Dùng cho evaluation multi-year, không dùng cho OOF trong training.

### Code refs

`src/validation.py::compute_purged_train_indices`, `src/validation.py::PurgedEmbargoTimeSeriesSplit.split`, `src/validation.py::walk_forward_split`. Caller chính: `src/models.py::cross_validate_oof_probabilities`.

## Tham số quan trọng

| Tham số | Giá trị | Vị trí | Lý do |
|---|---|---|---|
| `CV_SPLITS` | $5$ | `src/config.py:17` | 5 fold — cân bằng giữa variance ước lượng OOF và kích thước mỗi fold |
| `EMBARGO_PCT` | $0.02$ | `src/config.py:18` | Embargo $\approx 2\% \cdot N$ nến $\approx 880$ giờ trên 5 năm, $\approx 170$ giờ trên 1 năm |
| `LABELING_HORIZON` | $24$ | `src/config.py:53` | Khoảng label overlap tối đa — purge window phải $\geq h$ |
| `n_windows` | $3$ | `PipelineConfig.n_windows` | Walk-forward windows (mặc định) |

Lưu ý: purge window thực tế không phải tham số hard-code mà được tính từ `event_end` tối đa trong train — thường xấp xỉ `LABELING_HORIZON` nhưng có thể lớn hơn nếu barrier detect sớm. Embargo là tham số độc lập, áp lên tất cả fold bất kể `event_end`.

## Kết quả thực nghiệm

Trên 12 tháng hourly ($N \approx 8\,500$), `CV_SPLITS=5`, `EMBARGO_PCT=0.02`:

| Fold | test_start | test_end | Train size (sau purge) | Test size | Embargo (giờ) |
|---|---|---|---|---|---|
| 0 | 1 416 | 2 832 | $\approx 1 416$ | 1 417 | 170 |
| 1 | 2 832 | 4 248 | $\approx 2 645$ | 1 417 | 170 |
| 2 | 4 248 | 5 664 | $\approx 4 060$ | 1 417 | 170 |
| 3 | 5 664 | 7 080 | $\approx 5 475$ | 1 417 | 170 |
| 4 | 7 080 | 8 500 | $\approx 6 890$ | 1 420 | — (fold cuối) |

Tổng sample thực tế train qua 5 fold $\approx 5 \times$ trung bình, đủ cho OOF probability của stacking meta-learner. Purge loại $\approx 100$–$200$ observation mỗi fold tùy event_end phân bổ.

## Tham khảo

- `\cite{de_prado_2018_afml}` — López de Prado, *Advances in Financial Machine Learning*, Wiley 2018, Ch. 7.
- `docs/13-methodology-purged-cv.md` — lý thuyết purged embargo CV.
- `docs/05-models-stacking.md` — caller `cross_validate_oof_probabilities`.
- `docs/01-data-pipeline.md` — temporal split 1-fold cùng nguyên lý.
- `docs/08-config.md` — bảng đầy đủ tham số.
