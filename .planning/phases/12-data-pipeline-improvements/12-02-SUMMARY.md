---
phase: 12-data-pipeline-improvements
plan: 02
subsystem: data
tags: [serialization, error-handling, pathlib]

# Dependency graph
requires:
  - phase: 12-01
    provides: DataLoader wrapper and SpectraData.from_raw()
provides:
  - SpectraData.to_pickle() for round-trip serialization
  - Improved error messages throughout data pipeline
affects: [13-masking-gui, 14-jupyter-roi]

# Tech tracking
tech-stack:
  added: []
  patterns: [descriptive-error-messages, path-validation-on-init]

key-files:
  created: []
  modified:
    - spectral_select/types.py
    - spectral_select/loader.py

key-decisions:
  - "Follow from_pickle() format exactly for to_pickle() output"
  - "Add path existence checks early (init/call time) for fast failure"
  - "Include directory contents in file-not-found errors"

patterns-established:
  - "Error messages include: what happened, what was found, what was expected, hint for fix"
  - "Validate paths at initialization time for early failure"

issues-created: []

# Metrics
duration: 5min
completed: 2026-01-20
---

# Phase 12 Plan 02: to_pickle and Error Handling Summary

**Round-trip serialization with to_pickle() method and descriptive error messages throughout the data pipeline**

## Performance

- **Duration:** 5 min
- **Started:** 2026-01-20T12:15:38Z
- **Completed:** 2026-01-20T12:20:49Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Added SpectraData.to_pickle() enabling save → reload round-trip
- Improved from_pickle() error messages with found keys listing and format hints
- Added path existence validation in from_raw() and DataLoader.__init__
- Enhanced ImageJ installation guidance in error messages
- Empty directory detection with helpful messaging

## Task Commits

Each task was committed atomically:

1. **Task 1: Add SpectraData.to_pickle() method** - `15191e5` (feat)
2. **Task 2: Improve error handling with descriptive messages** - `6ebb258` (fix)

## Files Created/Modified
- `spectral_select/types.py` - Added to_pickle(), improved from_pickle() and from_raw() error handling
- `spectral_select/loader.py` - Added path validation in __init__, improved error messages for missing files and ImageJ

## Decisions Made
- **to_pickle() format:** Output matches from_pickle() expected structure exactly (data/excitation_wavelengths keys) for full compatibility
- **Error message pattern:** Include what happened + what was found + what was expected + actionable hint
- **Path validation timing:** Validate existence at initialization/call time rather than deep in loading for faster failure

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## Next Phase Readiness
- Data pipeline now supports complete raw → processed → saved workflow
- Error messages provide actionable guidance for common issues
- Ready for GUI tools (Phase 13) and Jupyter widgets (Phase 14)

---
*Phase: 12-data-pipeline-improvements*
*Completed: 2026-01-20*
