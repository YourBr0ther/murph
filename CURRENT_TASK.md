# Current Task: None

## Status: Ready for Next Feature

## Previous Task Completed
Conditional + Loneliness Behaviors - 2024-12-09

## Next Feature Options (from PROGRESS.md)
1. Spatial navigation behaviors (use spatial map for exploration)

## Notes
Conditional and loneliness behaviors implementation is complete with:

### New Loneliness Behaviors (4 new)
- **sigh**: Express loneliness with sad sound/expression (triggered by "lonely")
- **mope**: Slow, sad movement when very lonely (triggered by "very_lonely")
- **perk_up_hopeful**: Quick hopeful look around for someone (triggered by "lonely")
- **seek_company**: Actively search for people, react with joy if found (triggered by "lonely"/"very_lonely")

### Conditional Behavior Trees (3 refactored)
Using py-trees Selector composite with memory=False for condition re-evaluation:

- **explore**: Now reacts when person appears mid-exploration (PersonDetectedCondition)
  - If person within 150cm: express happiness, greet, and stop
  - Otherwise: continue normal exploration

- **wander**: Now reacts when near edge (TriggerActiveCondition("near_edge"))
  - If near edge: alert expression, stop, back up, turn 180°
  - Otherwise: continue normal wandering

- **seek_company**: Conditional tree that celebrates finding someone
  - If person found: express happiness, play sound, approach
  - Otherwise: scan, move around, end sad

### Files Modified
- `server/cognition/behavior/actions.py` - Added "sigh" sound (1.5s duration)
- `server/cognition/behavior/behavior_registry.py` - Added 4 loneliness behaviors
- `server/cognition/behavior/trees.py` - Added 4 loneliness trees, refactored explore/wander/seek_company
- `tests/test_server/test_loneliness_behaviors.py` - 23 new tests
- `tests/test_server/test_behavior_executor.py` - Updated tree count assertion

### Test Coverage
- 23 new tests for loneliness behaviors and conditional trees
- 557 total tests passing

### Loneliness Trigger System
- `lonely` trigger activates after 5 minutes without interaction
- `very_lonely` trigger activates after 10 minutes without interaction
- Loneliness behaviors get opportunity bonus from these triggers
- Behaviors have cooldowns to prevent repetitive expression (60s-180s)
