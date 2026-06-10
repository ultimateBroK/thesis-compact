#!/usr/bin/env python3
"""Download tick data using dukascopy-python library.

Downloads historical tick data from Dukascopy for any supported instrument,
saving as monthly parquet files with verify & repair support.

Defaults are resolved from ``config.toml`` when available.

Usage:
    pixi run python data_download.py
    pixi run python data_download.py --workers 4
    pixi run python data_download.py --force
    pixi run python data_download.py --no-verify
    pixi run python data_download.py --instrument XAG/USD --asset-class fx
"""

from __future__ import annotations

import argparse
import calendar
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAU_USD

try:
    from thesis.shared.config import load_config
except ImportError:
    load_config = None

logger = logging.getLogger(__name__)

INTERVAL_TICK = dukascopy_python.INTERVAL_TICK
OFFER_SIDE_BID = dukascopy_python.OFFER_SIDE_BID
STATE_FILE = "download_state.json"

COLUMN_RENAMES = {
    "timestamp": "timestamp",
    "askPrice": "ask",
    "bidPrice": "bid",
    "askVolume": "ask_volume",
    "bidVolume": "bid_volume",
}
COLUMN_ORDER = ["timestamp", "ask", "bid", "ask_volume", "bid_volume"]
TIMESTAMP_TYPE = pl.Datetime("us", "UTC")


def load_state(state_path: Path) -> dict[str, dict[str, int]]:
    if state_path.exists() and state_path.stat().st_size > 0:
        try:
            return json.loads(state_path.read_text())
        except json.JSONDecodeError:
            logger.warning("State file %s is corrupted, starting fresh", state_path)
    return {}


def save_state(state_path: Path, state: dict[str, dict[str, int]]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True))


def normalize_tick_dataframe(df_pandas) -> pl.DataFrame:
    df = pl.from_pandas(df_pandas.reset_index())
    return (
        df.rename(COLUMN_RENAMES)
        .select(COLUMN_ORDER)
        .with_columns(pl.col("timestamp").cast(TIMESTAMP_TYPE))
        .sort("timestamp")
    )


def ensure_utc_timestamp(df: pl.DataFrame) -> pl.DataFrame:
    if df.schema["timestamp"] != TIMESTAMP_TYPE:
        df = df.with_columns(pl.col("timestamp").cast(TIMESTAMP_TYPE))
    return df


def _trading_hour_slots(
    year: int, month: int, asset_class: str = "fx"
) -> list[tuple[int, int]]:
    days_in_month = calendar.monthrange(year, month)[1]
    if asset_class == "crypto":
        return [(d, h) for d in range(1, days_in_month + 1) for h in range(24)]

    slots: list[tuple[int, int]] = []
    for d in range(1, days_in_month + 1):
        wd = calendar.weekday(year, month, d)
        if wd == 5:
            continue
        hours = range(21, 24) if wd == 6 else range(24)
        slots.extend((d, h) for h in hours)
    return slots


def _covered_hour_slots(df: pl.DataFrame) -> set[tuple[int, int]]:
    if len(df) == 0:
        return set()
    return set(
        df.with_columns(
            pl.col("timestamp").dt.day().alias("_d"),
            pl.col("timestamp").dt.hour().alias("_h"),
        )
        .select(["_d", "_h"])
        .unique()
        .rows()
    )


def find_missing_hours(
    df: pl.DataFrame, year: int, month: int, asset_class: str
) -> list[tuple[int, int]]:
    expected = set(_trading_hour_slots(year, month, asset_class))
    return sorted(expected - _covered_hour_slots(df))


def _fetch_hour(
    instrument: str, year: int, month: int, day: int, hour: int, max_retries: int
) -> pl.DataFrame | None:
    start = datetime(year, month, day, hour, tzinfo=ZoneInfo("UTC"))
    if hour < 23:
        end = datetime(year, month, day, hour + 1, tzinfo=ZoneInfo("UTC"))
    elif month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=ZoneInfo("UTC"))
    else:
        end = datetime(year, month + 1, 1, tzinfo=ZoneInfo("UTC"))

    try:
        df_pandas = dukascopy_python.fetch(
            instrument,
            INTERVAL_TICK,
            OFFER_SIDE_BID,
            start,
            end,
            max_retries=max_retries,
            limit=30_000_000,
            debug=False,
        )
        if df_pandas is not None and len(df_pandas) > 0:
            return normalize_tick_dataframe(df_pandas)
    except Exception as e:
        logger.debug(
            "  Fetch failed %04d-%02d-%02d %02d:00: %s", year, month, day, hour, e
        )
    return None


def repair_missing_hours(
    df: pl.DataFrame,
    year: int,
    month: int,
    asset_class: str,
    instrument: str,
    max_retries: int,
) -> tuple[pl.DataFrame, int, int]:
    missing = find_missing_hours(df, year, month, asset_class)
    if not missing:
        return df, 0, 0

    logger.info("  Repairing %d missing hour slots...", len(missing))

    patch_frames: list[pl.DataFrame] = []
    still_missing = 0

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(
                _fetch_hour, instrument, year, month, day, hour, max_retries
            ): (day, hour)
            for day, hour in missing
        }
        for future in as_completed(futures):
            day, hour = futures[future]
            try:
                patch = future.result()
                if patch is not None:
                    patch_frames.append(patch)
                else:
                    still_missing += 1
            except Exception:
                still_missing += 1

    if not patch_frames:
        return df, 0, still_missing

    merged = (
        pl.concat([df] + patch_frames, how="diagonal")
        .unique(subset=["timestamp"], keep="first")
        .sort("timestamp")
    )
    return merged, len(merged) - len(df), still_missing


def _month_start(year: int, month: int) -> datetime:
    return datetime(year, month, 1, tzinfo=ZoneInfo("UTC"))


def _month_end(year: int, month: int) -> datetime:
    if month == 12:
        return datetime(year + 1, 1, 1, tzinfo=ZoneInfo("UTC"))
    return datetime(year, month + 1, 1, tzinfo=ZoneInfo("UTC"))


def _load_existing_parquet(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pl.read_parquet(path)
        return ensure_utc_timestamp(df)
    except Exception:
        logger.warning("Existing file %s is corrupted, re-downloading", path)
        return None


def _is_marked_complete(state_path: Path, key: str) -> bool:
    state = load_state(state_path)
    entry = state.get(key)
    return entry is not None and entry.get("missing_hours", -1) == 0


def _verify_and_repair(
    df: pl.DataFrame,
    year: int,
    month: int,
    asset_class: str,
    instrument: str,
    max_retries: int,
) -> tuple[pl.DataFrame, int, int]:
    missing_count = len(find_missing_hours(df, year, month, asset_class))
    if missing_count == 0:
        return df, 0, 0

    df, rows_added, still_missing = repair_missing_hours(
        df, year, month, asset_class, instrument, max_retries
    )
    if rows_added > 0:
        logger.info("  Repaired +%s rows", f"{rows_added:,}")
    return df, still_missing, still_missing


def download_month(
    year: int,
    month: int,
    output_dir: Path,
    instrument: str = INSTRUMENT_FX_METALS_XAU_USD,
    asset_class: str = "fx",
    max_retries: int = 7,
    force: bool = False,
    verify: bool = True,
) -> tuple[int, int, int]:
    key = f"{year}-{month:02d}"
    file_path = output_dir / f"{key}.parquet"
    state_path = output_dir / STATE_FILE
    now = datetime.now(timezone.utc)

    if _month_start(year, month) > now:
        logger.info("Skip     %s  (future)", key)
        return 0, 0, 0

    if not force and file_path.exists():
        if _is_marked_complete(state_path, key):
            rows = len(pl.read_parquet(file_path))
            logger.info("Skip     %s  rows=%10s  complete", key, f"{rows:,}")
            return rows, 0, 0

    df = _load_existing_parquet(file_path) if not force else None

    if df is None:
        start = _month_start(year, month)
        end = min(_month_end(year, month), now)
        logger.info("Download %s  (%s → %s)...", key, start.date(), end.date())

        try:
            df_pandas = dukascopy_python.fetch(
                instrument,
                INTERVAL_TICK,
                OFFER_SIDE_BID,
                start,
                end,
                max_retries=max_retries,
                limit=30_000_000,
                debug=False,
            )
            if df_pandas is None or len(df_pandas) == 0:
                logger.warning("No data returned for %s", key)
                return 0, 0, 0
            df = normalize_tick_dataframe(df_pandas)
        except Exception as e:
            logger.error("Failed to download %s: %s", key, e)
            return 0, 0, 0

    missing_hours, confirmed_absent = 0, 0
    if verify and len(df) > 0:
        df, missing_hours, confirmed_absent = _verify_and_repair(
            df, year, month, asset_class, instrument, max_retries
        )

    if len(df) > 0:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(file_path)
        rows = len(df)
    else:
        rows = 0

    flag = "complete" if missing_hours == 0 else f"{missing_hours} hrs missing"
    logger.info("   %s  rows=%10s  %s", key, f"{rows:,}", flag)

    if _month_end(year, month) < now:
        state = load_state(state_path)
        state[key] = {
            "rows": rows,
            "missing_hours": missing_hours,
            "confirmed_absent": confirmed_absent,
        }
        save_state(state_path, state)

    return rows, missing_hours, confirmed_absent


def _resolve_instrument(symbol: str) -> str:
    from dukascopy_python import instruments as instr_module

    clean = symbol.replace("/", "").upper()
    for attr_name in dir(instr_module):
        if not attr_name.startswith("INSTRUMENT"):
            continue
        value = getattr(instr_module, attr_name)
        if isinstance(value, str) and value.replace("/", "").upper() == clean:
            return value
    if clean.startswith("XAU"):
        return INSTRUMENT_FX_METALS_XAU_USD
    return symbol


def _load_download_config() -> dict | None:
    if load_config is None:
        return None
    try:
        cfg = load_config("config.toml")
    except FileNotFoundError:
        return None

    data_cfg = cfg.data
    start_dt = datetime.strptime(cfg.data_range.start, "%Y-%m-%d")
    end_dt = datetime.strptime(cfg.data_range.end, "%Y-%m-%d")
    return {
        "start_year": start_dt.year,
        "start_month": start_dt.month,
        "end_year": end_dt.year,
        "end_month": end_dt.month,
        "output_dir": Path(cfg.paths.data_raw),
        "instrument": _resolve_instrument(
            data_cfg.symbol_download or data_cfg.symbol
        ),
        "asset_class": data_cfg.asset_class,
        "max_retries": data_cfg.download_max_retries,
        "force": data_cfg.download_force,
        "skip_current_month": data_cfg.download_skip_current_month,
    }


def _builtin_defaults() -> dict:
    now = datetime.now(timezone.utc)
    return {
        "start_year": 2018,
        "start_month": 1,
        "end_year": now.year,
        "end_month": now.month,
        "output_dir": Path("data/XAUUSD"),
        "instrument": INSTRUMENT_FX_METALS_XAU_USD,
        "asset_class": "fx",
        "max_retries": 7,
        "force": False,
        "skip_current_month": True,
    }


def run_download(
    start_year: int | None = None,
    start_month: int | None = None,
    end_year: int | None = None,
    end_month: int | None = None,
    output_dir: Path | None = None,
    instrument: str | None = None,
    asset_class: str | None = None,
    workers: int = 4,
    max_retries: int | None = None,
    force: bool | None = None,
    verify: bool = True,
    skip_current_month: bool | None = None,
) -> bool:
    defaults = _load_download_config() or _builtin_defaults()

    start_year = start_year if start_year is not None else defaults["start_year"]
    start_month = start_month if start_month is not None else defaults["start_month"]
    end_year = end_year if end_year is not None else defaults["end_year"]
    end_month = end_month if end_month is not None else defaults["end_month"]
    output_dir = output_dir if output_dir is not None else defaults["output_dir"]
    instrument = instrument if instrument is not None else defaults["instrument"]
    asset_class = asset_class if asset_class is not None else defaults["asset_class"]
    max_retries = max_retries if max_retries is not None else defaults["max_retries"]
    force = force if force is not None else defaults["force"]
    skip_current_month = (
        skip_current_month
        if skip_current_month is not None
        else defaults["skip_current_month"]
    )

    symbol_dir = instrument.replace("/", "").upper()
    output_dir = Path("data") / symbol_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    months_to_download: list[tuple[int, int]] = []
    year, month = start_year, start_month
    while (year < end_year) or (year == end_year and month <= end_month):
        if skip_current_month and year == now.year and month == now.month:
            logger.info("Skip     %d-%02d  (current month)", year, month)
        else:
            months_to_download.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1

    logger.info(
        "Starting download of %d months to %s  [instrument=%s, asset_class=%s, workers=%d]",
        len(months_to_download),
        output_dir,
        instrument,
        asset_class,
        workers,
    )

    total_rows, total_missing, total_errors = 0, 0, 0

    if workers == 1:
        for y, m in months_to_download:
            rows, missing, _ = download_month(
                y, m, output_dir, instrument, asset_class, max_retries, force, verify
            )
            total_rows += rows
            total_missing += missing
            if rows == 0 and missing > 0:
                total_errors += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    download_month,
                    y,
                    m,
                    output_dir,
                    instrument,
                    asset_class,
                    max_retries,
                    force,
                    verify,
                ): (y, m)
                for y, m in months_to_download
            }
            for future in as_completed(futures):
                y, m = futures[future]
                try:
                    rows, missing, _ = future.result()
                    total_rows += rows
                    total_missing += missing
                    if rows == 0 and missing > 0:
                        total_errors += 1
                except Exception as e:
                    logger.error("Exception for %d-%02d: %s", y, m, e)
                    total_errors += 1

    logger.info(
        "Download complete: %s total rows, %d missing hours, %d errors",
        f"{total_rows:,}",
        total_missing,
        total_errors,
    )
    return total_errors == 0


def main(argv: list[str] | None = None) -> int:
    defaults = _load_download_config() or _builtin_defaults()

    parser = argparse.ArgumentParser(
        description="Download tick data from Dukascopy via dukascopy-python",
    )
    parser.add_argument(
        "--start-year", type=int, default=defaults["start_year"],
    )
    parser.add_argument(
        "--start-month", type=int, default=defaults["start_month"],
    )
    parser.add_argument(
        "--end-year", type=int, default=defaults["end_year"],
    )
    parser.add_argument(
        "--end-month", type=int, default=defaults["end_month"],
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--instrument", default=None)
    parser.add_argument(
        "--asset-class", choices=["fx", "crypto", "index"], default=None,
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument(
        "--force", action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument(
        "--skip-current-month",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    success = run_download(
        start_year=args.start_year,
        start_month=args.start_month,
        end_year=args.end_year,
        end_month=args.end_month,
        output_dir=args.output_dir,
        instrument=args.instrument,
        asset_class=args.asset_class,
        workers=args.workers,
        max_retries=args.max_retries,
        force=args.force,
        verify=not args.no_verify,
        skip_current_month=args.skip_current_month,
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

