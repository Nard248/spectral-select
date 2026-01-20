---
phase: 12-data-pipeline-improvements
plan: 01
subsystem: data
tags: [dataloader, im3, hyperspectral, imagej, pyimagej]

# Dependency graph
requires:
  - phase: 03-core-data-types
    provides: SpectraData, ExcitationData, LoadingOptions dataclasses
provides:
  - DataLoader class for loading raw .im3 files
  - SpectraData.from_raw() factory method
  - DataLoadingError exception for error handling
affects: [13-masking-gui-tool, data-preparation, raw-data-loading]

# Tech tracking
tech-stack:
  added: []
  patterns: [lazy initialization, wrapper pattern, exception chaining]

key-files:
  created: [spectral_select/loader.py]
  modified: [spectral_select/__init__.py, spectral_select/types.py]

key-decisions:
  - "Wrap HyperspectralDataLoader rather than duplicating code"
  - "Lazy ImageJ initialization only when actually loading .im3 files"
  - "DataLoadingError captures original exception as cause for debugging"

patterns-established:
  - "Lazy imports inside methods to avoid circular dependencies"
  - "Exception classes with path and cause attributes for context"

issues-created: []

# Metrics
duration: 4min
completed: 2026-01-20
---

# Phase 12 Plan 01: DataLoader Wrapper and SpectraData.from_raw() Summary

**DataLoader wrapper class with lazy ImageJ initialization and SpectraData.from_raw() factory method for loading raw .im3 hyperspectral files**

## Performance

- **Duration:** 4 min
- **Started:** 2026-01-20T11:55:51Z
- **Completed:** 2026-01-20T11:59:31Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- DataLoader class wrapping HyperspectralDataLoader with clean API
- Lazy ImageJ initialization to avoid startup delays when not loading .im3
- DataLoadingError exception with path and cause attributes for debugging
- SpectraData.from_raw() factory method for direct .im3 loading
- Graceful handling when pyimagej not installed

## Task Commits

Each task was committed atomically:

1. **Task 1: Create DataLoader wrapper class** - `5ecafce` (feat)
2. **Task 2: Add SpectraData.from_raw() class method** - `25a8f14` (feat)

**Plan metadata:** (pending)

## Files Created/Modified

- `spectral_select/loader.py` - New DataLoader class and DataLoadingError exception
- `spectral_select/__init__.py` - Export DataLoader and DataLoadingError
- `spectral_select/types.py` - Add from_raw() class method to SpectraData

## Decisions Made

- Wrap existing HyperspectralDataLoader rather than duplicating its code - avoids maintenance burden
- Use lazy ImageJ initialization - only import/init pyimagej when actually calling load()
- Exception chaining with cause attribute - preserves full error context for debugging

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## Next Phase Readiness

- DataLoader ready for use in data preparation workflows
- from_raw() provides clean API for loading raw data
- Ready for remaining Phase 12 plans (if any) or Phase 13

---
*Phase: 12-data-pipeline-improvements*
*Completed: 2026-01-20*
