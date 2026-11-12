from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from . import CONFIG


def load(path: Path | str = CONFIG) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)
