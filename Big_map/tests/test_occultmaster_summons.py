from summon_actions import handle_occultmaster_action


class _FakeBattle:
    def __init__(self) -> None:
        self.combined = [
            {
                "name": "Оккультмастер",
                "team": "red",
                "position": 1,
                "stand": "ahead",
                "health": 100,
                "max_health": 100,
                "big": False,
            }
        ]
        self.logs = []

    def _unit_by_position(self, position: int):
        for unit in self.combined:
            if int(unit.get("position", -1)) == int(position):
                return unit
        return None

    def _alive(self, unit: dict) -> bool:
        return int(unit.get("health", 0) or 0) > 0

    def _log(self, message: str) -> None:
        self.logs.append(message)


def test_occultmaster_summoned_wight_uses_battle_unit_type(monkeypatch):
    battle = _FakeBattle()
    attacker = battle.combined[0]

    monkeypatch.setattr("summon_actions.random.choice", lambda variants: variants[-1])

    spawned, reason = handle_occultmaster_action(
        battle,
        attacker,
        red_front_positions=[1],
        red_back_positions=[4],
        blue_front_positions=[7],
        blue_back_positions=[10],
    )

    summoned = battle._unit_by_position(4)
    assert spawned is True
    assert reason == "ok"
    assert summoned["name"] == "Сущий"
    assert summoned["unit_type"] == "Wight"
