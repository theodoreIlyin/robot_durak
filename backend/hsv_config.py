"""Завантаження та збереження HSV-діапазонів кольорів для автопілота."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable

from config.settings import DEFAULT_COLOR_HSV_RANGES, HSV_CONFIG_FILENAME

__all__ = [
    "ColorRange",
    "load_color_ranges",
    "save_color_ranges",
]


@dataclass
class ColorRange:
    name: str
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]


def _config_path() -> Path:
    """Повернути повний шлях до JSON-файла з HSV-діапазонами."""
    # config/ поруч із цим модулем.
    config_dir = Path(__file__).resolve().parent.parent / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / HSV_CONFIG_FILENAME


def load_color_ranges() -> Dict[str, ColorRange]:
    """Завантажити HSV-діапазони з JSON або створити їх за замовчуванням.

    Повертає словник name -> ColorRange.
    """
    path = _config_path()
    if not path.exists():
        ranges = {
            name: ColorRange(name=name, lower=tuple(defn["lower"]), upper=tuple(defn["upper"]))
            for name, defn in DEFAULT_COLOR_HSV_RANGES.items()
        }
        save_color_ranges(ranges.values())
        return ranges

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        result: Dict[str, ColorRange] = {}
        for item in data.get("colors", []):
            name = str(item["name"])
            raw_lower = tuple(int(x) for x in item["lower"])
            raw_upper = tuple(int(x) for x in item["upper"])
            if len(raw_lower) != 3 or len(raw_upper) != 3:
                continue
            # H: 0–179, S і V: 0–255 — відповідно до діапазонів OpenCV.
            h_max, sv_max = 179, 255
            limits = ((0, h_max), (0, sv_max), (0, sv_max))
            lower = tuple(max(lo, min(hi, v)) for (lo, hi), v in zip(limits, raw_lower))
            upper = tuple(max(lo, min(hi, v)) for (lo, hi), v in zip(limits, raw_upper))
            # Пропускаємо запис якщо після затискання мін перевищує макс.
            if any(lo > hi for lo, hi in zip(lower, upper)):
                continue
            result[name] = ColorRange(name=name, lower=lower, upper=upper)
        if not result:
            raise ValueError("empty color ranges")
        return result
    except Exception:
        # У разі пошкодженого файла повертаємо дефолтні значення.
        ranges = {
            name: ColorRange(name=name, lower=tuple(defn["lower"]), upper=tuple(defn["upper"]))
            for name, defn in DEFAULT_COLOR_HSV_RANGES.items()
        }
        save_color_ranges(ranges.values())
        return ranges


def save_color_ranges(ranges: Iterable[ColorRange]) -> None:
    """Зберегти HSV-діапазони в JSON-конфіг (перезапис)."""
    path = _config_path()
    serializable = {
        "colors": [
            {
                "name": r.name,
                "lower": list(r.lower),
                "upper": list(r.upper),
            }
            for r in ranges
        ]
    }
    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")