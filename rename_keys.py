from pathlib import Path


def main() -> None:
    path = Path("1workingbattle.py")
    text = path.read_text(encoding="utf-8")

    mapping = {
        "инициатива_база": "initiative_base",
        "инициатива": "initiative",
        "имя": "name",
        "Урон": "damage",
        "урон2": "damage_secondary",
        "Здоровье": "health",
        "maxhealth": "max_health",
        "броня": "armor",
        "Точность2": "accuracy_secondary",
        "Точность": "accuracy",
        "иммунитет": "immunity",
        "Стойкость": "resistance",
        "Тип атаки 1": "attack_type_primary",
        "Тип атаки 2": "attack_type_secondary",
        "Type": "unit_type",
        "paralized": "paralyzed",
        "longparalized": "long_paralyzed",
        "Runningaway": "running_away",
    }

    # Replace longer keys first to avoid partial overlaps (e.g., Точность2 vs Точность).
    for old in sorted(mapping.keys(), key=len, reverse=True):
        new = mapping[old]
        for quote in ('"', "'"):
            text = text.replace(f"{quote}{old}{quote}", f"{quote}{new}{quote}")

    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

