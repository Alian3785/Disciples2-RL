# maps/__init__.py
"""Реестр кампанейских карт.

Каждая карта — модуль в этом пакете, экспортирующий константу MAP: MapConfig.
Выбор карты: CampaignEnv(map_name=...) или CLI-флаг --map.
"""

from __future__ import annotations

import importlib
from functools import lru_cache

from maps.base import MapConfig

_REGISTRY = {
    "default": "maps.default",
    "orc_duel": "maps.orc_duel",
    "small": "maps.small",
}


def available_maps() -> tuple[str, ...]:
    """Имена всех зарегистрированных карт."""
    return tuple(_REGISTRY.keys())


@lru_cache(maxsize=None)
def get_map(name: str = "default") -> MapConfig:
    """Возвращает конфиг карты по имени (кешируется)."""
    normalized = str(name or "default").strip().lower()
    module_path = _REGISTRY.get(normalized)
    if module_path is None:
        raise ValueError(
            f"Unknown map {name!r}; available maps: {', '.join(available_maps())}"
        )
    module = importlib.import_module(module_path)
    map_config = getattr(module, "MAP", None)
    if not isinstance(map_config, MapConfig):
        raise ValueError(f"Map module {module_path} does not export MAP: MapConfig")
    return map_config
