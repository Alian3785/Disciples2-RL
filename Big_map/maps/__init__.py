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
    "builder": "maps.builder",
    "default": "maps.default",
    "formation_train": "maps.formation_train",
    "hire_train": "maps.hire_train",
    "item_train": "maps.item_train",
    "magic_train": "maps.magic_train",
    "orc_duel": "maps.orc_duel",
    "scroll_train": "maps.scroll_train",
    "siege_train": "maps.siege_train",
    "small": "maps.small",
    "super_last_stand": "maps.super_last_stand",
    "trade_train": "maps.trade_train",
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
