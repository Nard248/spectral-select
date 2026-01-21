---
phase: 13-masking-gui-tool
plan: 03
subsystem: ui
tags: [tkinter, matplotlib, spectral-analysis, histogram, statistics]

# Dependency graph
requires:
  - phase: 13-02
    provides: ViewerApp with band browser, false color, zoom controls
provides:
  - Spectral profile panel with click-to-plot extraction
  - Multi-point spectrum comparison (up to 5 traces)
  - Histogram visualization with log scale and percentile markers
  - Image statistics panel (min/max/mean/median/std/count)
  - ROI statistics when mask exists
  - View menu for panel visibility toggles
affects: [13-04 (mask operations), visualization workflows]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Module-level helper functions for testability (extract_spectrum, compute_statistics)
    - View menu with checkbutton panel toggles
    - Statistics caching pattern for performance

key-files:
  created: []
  modified:
    - spectral_select/viewer.py

key-decisions:
  - "extract_spectrum() returns cube[y, x, :] copy for safe modification"
  - "compute_image_statistics() handles NaN values gracefully"
  - "Histogram uses 256 bins with 2%/98% percentile markers"
  - "Statistics panel supports 'Current View' vs 'All Bands' mode toggle"
  - "ROI statistics section only shown when mask exists"

patterns-established:
  - "Panel visibility via View menu checkbuttons"
  - "Module-level helper functions for core operations"
  - "Statistics/histogram update triggered by _update_display()"

issues-created: []

# Metrics
duration: 12min
completed: 2026-01-21
---

# Phase 13-03: Spectral Profile and Statistics Summary

**Spectral profile click-to-plot, histogram visualization with percentile markers, and statistics panel with ROI support**

## Performance

- **Duration:** 12 min
- **Started:** 2026-01-21T18:17:48Z
- **Completed:** 2026-01-21T18:29:58Z
- **Tasks:** 3
- **Files modified:** 1

## Accomplishments
- Click on any pixel extracts and plots full spectrum with wavelength x-axis
- Compare mode overlays up to 5 spectra from different pixels with legend
- Histogram panel shows intensity distribution with 2%/98% percentile markers
- Statistics panel displays min/max/mean/median/std/count for current view
- ROI statistics section appears when mask exists in data
- View menu toggles visibility of all analysis panels

## Task Commits

Each task was committed atomically:

1. **Task 1: Add spectral profile panel with click-to-plot** - `a161781` (feat)
2. **Task 2: Add histogram and statistics panel** - `4673da1` (feat)
3. **Task 3: Integrate panels with responsive layout** - `ff43936` (feat)

**Plan metadata:** (pending)

## Files Created/Modified
- `spectral_select/viewer.py` - Added spectral profile panel, histogram panel, statistics panel, View menu, helper functions

## Decisions Made
- Module-level helper functions (extract_spectrum, compute_image_statistics, compute_histogram) for testability
- Statistics panel uses toggle between "Current View" and "All Bands" modes
- ROI statistics section is conditionally displayed only when mask exists in SpectraData
- Panel visibility state stored in BooleanVar for menu checkbutton sync

## Deviations from Plan

None - plan executed exactly as written

## Issues Encountered
None

## Next Phase Readiness
- Spectral profile and statistics panels complete
- Phase 13 complete - ME-HSI Viewer has core viewing, browsing, and analysis features
- Ready for Phase 14: Jupyter ROI Widget

---
*Phase: 13-masking-gui-tool*
*Completed: 2026-01-21*
