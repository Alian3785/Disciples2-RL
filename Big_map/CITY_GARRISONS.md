# City garrisons

Each non-capital settlement has five permanent action slots, in
`garrison_city_names` order, using that faction's existing `active_hire_options`.
The block begins at `GRID_GARRISON_HIRE_ACTION_START`. Capturing a city enables
its eligible actions; losing it disables all five. Recruitment is remote and
does not move the hero or change the travelling party.

A city of level 1, 2, 3, 4 or 5 has respectively 1, 2, 3, 4 or 5 reserve places.
Small units consume one; large units consume two and require both positions
in a free front/back column. Warriors randomly take a free front position
(7, 8, 9), mages/support/archers a free back position (10, 11, 12); large units
randomly take a free column. Placement uses the environment's seeded RNG.
Gold costs and capital building requirements are shared with faction hiring.
Recruitment has no separate reward and does not consume hero movement.

Rules references: the [official Disciples II manual, Cities](https://manuals.plus/m/ac61986f73b2b5016fec8ab1d17d7874a84463082bbb10d7dd6c1c3d3a0557a9)
describes level-limited reserves; [City Growth in the Disciples II FAQ](https://gamefaqs.gamespot.com/pc/926265-disciples-ii-gold-edition/faqs/23575)
lists the 1–5 capacities explicitly.

The existing scripted capital bot can target player cities. It must enter the
entrance tile (adjacency alone does not attack). Movement stops there, and the
next safe transition to battle gives BLUE control to the agent's city guards.
If a hero battle is already active, the city defence waits for its completion.
Defenders receive the existing level-dependent city armour bonus without the
hero's equipment. Survivors keep HP and XP; temporary battle effects and city
armour are removed when saved. Dead guards are removed after battle.

Living, wounded guards regenerate automatically once per campaign day, before
the scripted bot moves. The Disciples II fort formula is **innate unit regen +
city bonus + warrior-lord bonus**; the friendly-terrain bonus applies only to
units outside fortifications and is not added to guards. The current recruitable
units have 5% innate regeneration (a scenario can override `regeneration_percent`
in percentage points). City bonuses for levels 1–5 are 10/15/20/25/30%; warrior
lords (`typeoflord == 1`) add 15 percentage points. Thus ordinary guards restore
15/20/25/30/35% of maximum HP daily, or 30/35/40/45/50% under a warrior lord.

The rate is capped at 100%, HP at the unit's maximum; dead units are not revived.
HP uses the campaign's existing fractional representation. Each elapsed day is
processed separately, and an upgrade changes the next day's city bonus. Healing
does not cost gold, require a Temple or the hero's presence, or award a reward.
Hero equipment does not affect a separate garrison. Lost cities and units in a
pending/active defence do not heal. Updated HP appears in the existing garrison
observation block without changing its size.

Formula verified against the reverse-engineered game implementation in
[D2ModdingToolset, getUnitRegen/getFortRegen](https://github.com/VladimirMakeev/D2ModdingToolset/blob/10b24c6b71a23074a858e854035257b21afbaf32/mss32/src/unitutils.cpp#L359-L474).
The [official DII unit tables](https://manuals.plus/m/ac61986f73b2b5016fec8ab1d17d7874a84463082bbb10d7dd6c1c3d3a0557a9.pdf)
list the recruits' base regeneration; city bonuses are listed in the FAQ above.

On defeat the bot takes ownership, city recruitment/upgrades are masked, and
the city spreads the bot side's faction terrain at its level's growth rate.
The existing `empire_territory_*` fields represent this opponent side, as they
already do on maps whose scripted bot is another faction. The hero and campaign
remain alive. Empty cities fall immediately. A cleared city can subsequently
be recaptured by entering it. On victory the bot follows its existing respawn
cycle. On combat timeout surviving guards stay in the city and the bot returns.
The feature does not enable the bot by default.

Observations append 23 values per city, without moving the existing encoder
slices: player ownership, known bot ownership, owned-city capacity / 5, occupied
places / 5, active defence flag, then six triples of recruit type / 5, HP ratio,
and large-unit flag. Only player-owned garrisons expose their roster. The block
is available through `garrison_obs_slice`; snapshots expose `city_garrisons`
and `city_owners`. New actions preserve all prior action indices.

**Checkpoint compatibility:** on maps with cities, both action and observation
widths increase. Start new policies or explicitly migrate weights before using
an older checkpoint. Existing observation encodings of individual branches
remain in place before the appended block.

Run regression tests with `PYTHONPATH=. python -m pytest -q tests/test_city_garrisons.py`.
Tests use ordinary starting rosters and explicitly disable the bot except in
focused city-defence unit tests. They do not start training or evaluation runs.

City guards cannot select Retreat. The travelling hero and party also cannot
select Retreat when the battle starts on a city or capital entrance tile.
Neighbouring tiles, including owned territory, do not impose this restriction.
The origin of the hero party is used, not the enemy destination tile; a remote
summoned army does not inherit the restriction from the hero’s location.
The final siege army retains its separate retreat prohibition. Illegal direct
Retreat actions are rejected as well as excluded from the action mask.
