"""Phase 0 stub generator for thesis documentation tree.

Generates 21 markdown stubs + 1 BibTeX file + 1 README.md into ``docs/``
relative to the repo root. Idempotent: existing files >100 bytes are
skipped so the script can be re-run safely.

Run from repo root::

    python tools/gen_doc_stubs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_DOCS = Path(__file__).resolve().parent.parent / "docs"
LAST_UPDATED = "2026-06-02"
PLACEHOLDER = "[Chưa viết]"
DEFAULT_SECTIONS: list[str] = [
    "Tóm tắt",
    "Cơ sở lý thuyết",
    "Công thức",
    "Cài đặt",
    "Tham số quan trọng",
    "Kết quả thực nghiệm",
    "Tham khảo",
]


DOCS: list[dict] = [
    {
        "filename": "00-reproducibility.md",
        "title": "Reproducibility",
        "summary": "Môi trường Pixi, seed control, data acquisition, hardware spec phục vụ tái lập kết quả.",
        "stage": "reproducibility",
        "thesis_chapter": "A",
    },
    {
        "filename": "01-data-pipeline.md",
        "title": "Data Pipeline",
        "summary": "Parquet → OHLC Polars streaming, timeframe 1h, cấu trúc dữ liệu XAU/USD.",
        "stage": "data",
        "thesis_chapter": 3,
    },
    {
        "filename": "02-features.md",
        "title": "Feature Engineering",
        "summary": "21 đặc trưng đầu vào cho stacking ensemble: fractional differencing, chỉ báo kỹ thuật, OBV, wavelet.",
        "stage": "features",
        "thesis_chapter": 3,
    },
    {
        "filename": "03-labeling-triple-barrier.md",
        "title": "Labeling — Triple Barrier",
        "summary": "Swing H/L + ATR fallback + auto-tune barriers + meta-labeling cho position sizing.",
        "stage": "labeling",
        "thesis_chapter": 3,
    },
    {
        "filename": "04-validation-purged-embargo.md",
        "title": "Validation — Purged Embargo",
        "summary": "PurgedEmbargoTimeSeriesSplit, phòng tránh leakage trong time series tài chính.",
        "stage": "validation",
        "thesis_chapter": 3,
    },
    {
        "filename": "05-models-stacking.md",
        "title": "Models — Stacking Ensemble",
        "summary": "GRU + LightGBM + SVC + meta-learner + smart filtering — core contribution của luận văn.",
        "stage": "models",
        "thesis_chapter": 3,
    },
    {
        "filename": "06-backtest.md",
        "title": "Backtest",
        "summary": "Barrier-based equity sim, leverage, position sizing trên chuỗi trade giả lập.",
        "stage": "backtest",
        "thesis_chapter": 3,
    },
    {
        "filename": "07-reporting.md",
        "title": "Reporting",
        "summary": "reports/run_*/ artifacts, figures, JSON schema cho mỗi lần chạy pipeline.",
        "stage": "reporting",
        "thesis_chapter": 3,
    },
    {
        "filename": "08-config.md",
        "title": "Configuration",
        "summary": "Toàn bộ hyperparams trong src/config.py kèm rationale cho từng tham số.",
        "stage": "config",
        "thesis_chapter": 3,
    },
    {
        "filename": "09-cli.md",
        "title": "CLI Orchestration",
        "summary": "CLI orchestration qua main.py, flow subcommand, exit codes, logging.",
        "stage": "cli",
        "thesis_chapter": 3,
    },
    {
        "filename": "10-methodology-fracdiff.md",
        "title": "Methodology — Fractional Differencing",
        "summary": "Fractional differencing (0<d<1): toán lý thuyết và lý do chọn d=0.4.",
        "stage": "methodology",
        "thesis_chapter": 2,
        "skip_sections": ["Cài đặt"],
    },
    {
        "filename": "11-methodology-triple-barrier.md",
        "title": "Methodology — Triple Barrier",
        "summary": "Lý thuyết triple-barrier của López de Prado cho gán nhãn chuỗi thời gian tài chính.",
        "stage": "methodology",
        "thesis_chapter": 2,
        "skip_sections": ["Cài đặt"],
    },
    {
        "filename": "12-methodology-meta-labeling.md",
        "title": "Methodology — Meta-Labeling",
        "summary": "Meta-labeling nhằm cải thiện position sizing và giảm false positives.",
        "stage": "methodology",
        "thesis_chapter": 2,
        "skip_sections": ["Cài đặt"],
    },
    {
        "filename": "13-methodology-purged-cv.md",
        "title": "Methodology — Purged Cross-Validation",
        "summary": "Purged k-fold kèm embargo cho financial data, phòng tránh information leakage.",
        "stage": "methodology",
        "thesis_chapter": 2,
        "skip_sections": ["Cài đặt"],
    },
    {
        "filename": "14-methodology-stacking.md",
        "title": "Methodology — Stacking Ensemble",
        "summary": "Lý thuyết stacking ensemble và out-of-fold predictions cho kết hợp mô hình không đồng nhất.",
        "stage": "methodology",
        "thesis_chapter": 2,
        "skip_sections": ["Cài đặt"],
    },
    {
        "filename": "20-experiments.md",
        "title": "Experiments",
        "summary": "Hyperparameter grid, seed sweep, ablation design phục vụ so sánh mô hình.",
        "stage": "experiments",
        "thesis_chapter": 4,
    },
    {
        "filename": "21-results-convention.md",
        "title": "Results Convention",
        "summary": "Cách đọc reports/run_*, bảng kết quả chuẩn, ablation matrix cho luận văn.",
        "stage": "results",
        "thesis_chapter": 4,
    },
    {
        "filename": "22-evaluation-metrics.md",
        "title": "Evaluation Metrics",
        "summary": "F1, Sharpe, max-drawdown, win rate, PnL — định nghĩa và công thức tính.",
        "stage": "metrics",
        "thesis_chapter": 4,
    },
    {
        "filename": "30-exploratory-analysis.md",
        "title": "Exploratory Data Analysis",
        "summary": "Convert từ viz.ipynb sang markdown phục vụ trích dẫn trong luận văn.",
        "stage": "eda",
        "thesis_chapter": 4,
    },
    {
        "filename": "90-glossary.md",
        "title": "Glossary",
        "summary": "Thuật ngữ VI/EN: TP, SL, ATR, OOF, embargo, stacking, fractional diff, meta-labeling.",
        "stage": "glossary",
        "thesis_chapter": "B",
    },
]


BIBTEX_ENTRIES = """@book{de_prado_2018_afml,
  author    = {de Prado, Marcos Lopez},
  title     = {Advances in Financial Machine Learning},
  publisher = {Wiley},
  year      = {2018},
}

@book{hamilton_1994_time_series,
  author    = {Hamilton, James D.},
  title     = {Time Series Analysis},
  publisher = {Princeton University Press},
  year      = {1994},
}

@book{goodfellow_2016_dl,
  author    = {Goodfellow, Ian and Bengio, Yoshua and Courville, Aaron},
  title     = {Deep Learning},
  publisher = {MIT Press},
  year      = {2016},
}

@article{pedregosa_2011_sklearn,
  author  = {Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
             and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
             and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A.
             and Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  title   = {Scikit-learn: Machine Learning in {Python}},
  journal = {Journal of Machine Learning Research},
  volume  = {12},
  pages   = {2825--2830},
  year    = {2011},
}

@inproceedings{cho_2014_gru,
  author    = {Cho, Kyunghyun and van Merri{\"e}nboer, Bart and Gulcehre, Caglar
               and Bahdanau, Dzmitry and Bougares, Fethi and Schwenk, Holger
               and Bengio, Yoshua},
  title     = {Learning Phrase Representations using {RNN} Encoder--Decoder
               for Statistical Machine Translation},
  booktitle = {Proceedings of the 2014 Conference on Empirical Methods in
               Natural Language Processing (EMNLP)},
  year      = {2014},
}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def should_skip(path: Path) -> bool:
    """Return True iff target exists and is larger than 100 bytes."""
    return path.exists() and path.stat().st_size > 100


def build_frontmatter(entry: dict) -> str:
    """Render YAML frontmatter block for a doc entry."""
    doc_key = entry["filename"].removesuffix(".md")
    chapter = entry["thesis_chapter"]
    chapter_str = str(chapter) if isinstance(chapter, int) else chapter
    lines = [
        "---",
        f"doc: {doc_key}",
        f"stage: {entry['stage']}",
        f"thesis_chapter: {chapter_str}",
        f"status: draft",
        f"last_updated: {LAST_UPDATED}",
        f"code_ref: null",
        "---",
    ]
    return "\n".join(lines)


def build_markdown(entry: dict) -> str:
    """Render full markdown body (frontmatter + title + summary + sections)."""
    skip = set(entry.get("skip_sections") or [])
    parts: list[str] = [build_frontmatter(entry), ""]
    parts.append(f"# {entry['title']}")
    parts.append("")
    parts.append(f"> {entry['summary']}")
    parts.append("")
    for section in DEFAULT_SECTIONS:
        if section in skip:
            continue
        parts.append(f"## {section}")
        parts.append("")
        parts.append(PLACEHOLDER)
        parts.append("")
    # Trim trailing blank line for tidy output.
    while parts and parts[-1] == "":
        parts.pop()
    parts.append("")
    return "\n".join(parts)


def write_if_new(path: Path, content: str) -> bool:
    """Write ``content`` to ``path`` unless existing file is large enough.

    Returns True iff a new file was written.
    """
    if should_skip(path):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def chapter_label(chapter: int | str) -> str:
    """Human-readable chapter label for the README mapping table."""
    if isinstance(chapter, int):
        return f"Chương {chapter}"
    if chapter == "A":
        return "Phụ lục A — Môi trường thực nghiệm"
    if chapter == "B":
        return "Phụ lục B — Danh mục thuật ngữ"
    return str(chapter)


def build_readme() -> str:
    """Render docs/README.md content."""
    frontmatter = "\n".join(
        [
            "---",
            "thesis_title: [Chưa đặt tên]",
            "author: Nguyen Duc Hieu",
            "supervisor: [Chưa cung cấp]",
            "university: [Chưa cung cấp]",
            "department: [Chưa cung cấp]",
            f"last_updated: {LAST_UPDATED}",
            "status: draft",
            "---",
        ]
    )

    lines: list[str] = [frontmatter, ""]
    lines.append("# Tài liệu luận văn")
    lines.append("")
    lines.append(
        "> Bộ tài liệu phục vụ luận văn về pipeline stacking ensemble "
        "(GRU + LightGBM + SVC) dự báo XAU/USD với purged-embargo CV."
    )
    lines.append("")

    # Mapping table
    lines.append("## Ánh xạ tài liệu — chương luận văn")
    lines.append("")
    lines.append("| File | Chương |")
    lines.append("|---|---|")
    for entry in DOCS:
        lines.append(
            f"| `{entry['filename']}` | {chapter_label(entry['thesis_chapter'])} |"
        )
    lines.append(
        f"| `references.bib` | Nguồn cite duy nhất (IEEE numeric) |"
    )
    lines.append("")

    # Directory structure listing
    lines.append("## Cấu trúc thư mục")
    lines.append("")
    for entry in DOCS:
        lines.append(f"- `{entry['filename']}` — {entry['summary']}")
    lines.append("- `references.bib` — BibTeX seed (IEEE numeric), single source of truth.")
    lines.append("")

    # Conventions
    lines.append("## Quy ước viết")
    lines.append("")
    lines.append("- Ngôn ngữ chính: tiếng Việt; thuật ngữ kỹ thuật giữ tiếng Anh trong ngoặc.")
    lines.append("- Tên file: `kebab-case` với tiền tố số `NN-` đảm bảo thứ tự sort tự nhiên.")
    lines.append("- Không dùng suffix ngôn ngữ (`*.vi.md`); nếu cần bản EN thì tạo `docs-en/` riêng.")
    lines.append("- Citation style: IEEE numeric, cite từ `docs/references.bib` duy nhất.")
    lines.append("- Mỗi doc có YAML frontmatter với `doc`, `stage`, `thesis_chapter`, `status`, `last_updated`, `code_ref`.")
    lines.append("- Stub template: 7 section cố định (Tóm tắt → Cơ sở lý thuyết → Công thức → Cài đặt → Tham số quan trọng → Kết quả thực nghiệm → Tham khảo).")
    lines.append("- Methodology docs (10-14) bỏ section `Cài đặt` vì lý thuyết thuần.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Generate stubs and report write/skip counts."""
    targets: list[tuple[Path, str]] = []
    for entry in DOCS:
        path = REPO_DOCS / entry["filename"]
        targets.append((path, build_markdown(entry)))
    targets.append((REPO_DOCS / "references.bib", BIBTEX_ENTRIES))
    targets.append((REPO_DOCS / "README.md", build_readme()))

    written = 0
    skipped = 0
    for path, content in targets:
        if write_if_new(path, content):
            written += 1
        else:
            skipped += 1

    print(f"Wrote {written} new files, skipped {skipped} existing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
