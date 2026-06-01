from __future__ import annotations

from .args import derive_config_from_arguments, parse_command_line_arguments
from .pipeline import run_pipeline


def main() -> None:
    run_pipeline(derive_config_from_arguments(parse_command_line_arguments()))
