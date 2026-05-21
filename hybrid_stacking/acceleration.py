from __future__ import annotations

from accelerate import Accelerator
from accelerate.utils import set_seed


def configure_accelerator(random_state: int) -> Accelerator:
    set_seed(random_state)
    return Accelerator()
