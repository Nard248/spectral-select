---
phase: 03-core-data-types
plan: 01
subsystem: data
tags: [dataclass, numpy, hyperspectral, pickle, serialization]

# Dependency graph
requires:
  - phase: 02-config-system
    provides: Config dataclass pattern with validation and serialization
provides:
  - LoadingOptions for preprocessing configuration
  - ExcitationData for per-excitation spectral cubes
  - SpectraData container for multi-excitation hyperspectral data
affects: [04-analysis-engine, data-loading, visualization]

# Tech tracking
tech-stack:
  added: []
  patterns: [dataclass with __post_init__ validation, property-based accessors]

key-files:
  created: []
  modified: [spectral_select/types.py]

key-decisions:
  - "Exclude large arrays from to_dict() for lightweight serialization"
  - "Generate placeholder emission wavelengths when loading existing pkl format"
  - "Allow empty excitations dict for SpectraData initialization"

patterns-established:
  - "Property accessors for derived attributes (height, width, n_bands)"
  - "Separate to_dict() for metadata-only vs from_pickle() for full data"

issues-created: []

# Metrics
duration: 3min
completed: 2026-01-19
---

# Phase 3 Plan 01: Input Data Types Summary

**LoadingOptions, ExcitationData, and SpectraData dataclasses for typed hyperspectral data handling with validation and pickle format support**

## Performance

- **Duration:** 3 min
- **Started:** 2026-01-19T12:59:49Z
- **Completed:** 2026-01-19T13:03:02Z
- **Tasks:** 3
- **Files modified:** 1

## Accomplishments

- LoadingOptions dataclass capturing all preprocessing toggles (cutoff, normalization, ROI, downscaling)
- ExcitationData dataclass for per-excitation 3D spectral cubes with height/width/n_bands properties
- SpectraData container with from_pickle() supporting existing codebase pickle format
- Full validation in __post_init__ for all classes (parameter ranges, array shapes, spatial consistency)

## Task Commits

Each task was committed atomically:

1. **Task 1-3: All data types** - `aff84d7` (feat) - All three classes implemented in single commit

**Plan metadata:** (pending)

_Note: All three classes were implemented together as they share the same file and are interdependent_

## Files Created/Modified

- `spectral_select/types.py` - Added LoadingOptions, ExcitationData, SpectraData dataclasses

## Decisions Made

- **Lightweight to_dict():** SpectraData.to_dict() excludes large numpy arrays (cube, mask) to keep output lightweight; use pickle for full data serialization
- **Placeholder emission wavelengths:** When loading existing pkl format, emission wavelengths are generated as placeholder range(n_bands) since the format doesn't store per-band wavelengths
- **Empty initialization:** SpectraData allows empty excitations dict for flexibility in construction

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## Next Phase Readiness

- Data type foundation complete for Phase 4 (Analysis Engine)
- SpectraData.from_pickle() ready to load existing processed data
- Ready for 03-02: Result classes for wavelength selection outputs

---
*Phase: 03-core-data-types*
*Completed: 2026-01-19*
