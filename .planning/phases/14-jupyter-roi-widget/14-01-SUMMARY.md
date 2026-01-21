---
phase: 14-jupyter-roi-widget
plan: 01
subsystem: ui
tags: [ipympl, ipywidgets, matplotlib, jupyter, roi, widgets]

# Dependency graph
requires:
  - phase: 13-masking-gui-tool
    provides: create_rgb_image pattern, viewer architecture
provides:
  - ROIWidget class for Jupyter notebook ROI selection
  - create_display_image() helper for cube visualization
  - path_to_mask() helper for lasso-to-mask conversion
  - Rectangle and lasso selection tools
  - Copy-pasteable ROI coordinate extraction
affects: [14-02-multi-class-roi, ground-truth-creation]

# Tech tracking
tech-stack:
  added: [ipympl]
  patterns: [ipywidgets-output-capture, matplotlib-selector-widgets]

key-files:
  created: [spectral_select/widgets.py]
  modified: [spectral_select/__init__.py, requirements.txt]

key-decisions:
  - "Rectangle selector as default tool (more common use case)"
  - "get_roi_code() provides copy-pasteable Python for reproducibility"
  - "Store bounds tuple (row_min, row_max, col_min, col_max) for array slicing"
  - "Support both rectangle and lasso tools via tool parameter"

patterns-established:
  - "ipywidgets Output context manager for matplotlib figure capture"
  - "RectangleSelector with interactive=True for resizable selections"

issues-created: []

# Metrics
duration: 8min
completed: 2026-01-21
---

# Phase 14 Plan 01: ROIWidget Core Summary

**ipympl-based ROIWidget with rectangle/lasso selectors and copy-pasteable coordinate extraction for Jupyter notebooks**

## Performance

- **Duration:** 8 min (execution time, excludes user verification)
- **Started:** 2026-01-21T07:46:22Z
- **Completed:** 2026-01-21T13:11:18Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- ROIWidget class with interactive ROI selection in Jupyter notebooks
- Rectangle selector (default) for square/rectangular ROI drawing
- Lasso selector option for freeform ROI selection
- get_bounds() returns (row_min, row_max, col_min, col_max) for array slicing
- get_roi_code() and print_roi_code() for copy-pasteable Python code
- Semi-transparent red overlay visualization of selected region
- create_display_image() helper for cube-to-2D visualization
- path_to_mask() helper for converting lasso vertices to binary mask

## Task Commits

Each task was committed atomically:

1. **Task 1: Add ipympl dependency** - `3f8fce2` (chore)
2. **Task 2: Create ROIWidget with LassoSelector** - `8113ee9` (feat)
3. **Task 2b: Add rectangle selector and coordinate extraction** - `6b50385` (feat)

**Plan metadata:** (pending docs commit)

## Files Created/Modified

- `spectral_select/widgets.py` - New widget module with ROIWidget class and helpers
- `spectral_select/__init__.py` - Export ROIWidget, create_display_image, path_to_mask
- `requirements.txt` - Add ipympl dependency

## Decisions Made

- Rectangle selector as default (tool="rectangle") since rectangular ROIs are more common for region selection
- Bounds stored as (row_min, row_max, col_min, col_max) tuple matching numpy slice convention
- get_roi_code() generates complete copy-pasteable code including slice and mask examples
- Support both tools via parameter rather than separate widget classes

## Deviations from Plan

### User-Requested Enhancements

**1. Added rectangle selector tool**
- **Requested during:** Checkpoint verification
- **Reason:** User needed square ROI selection, not just freeform lasso
- **Implementation:** Added RectangleSelector with interactive=True for resizable selections
- **Committed in:** 6b50385

**2. Added coordinate extraction methods**
- **Requested during:** Checkpoint verification
- **Reason:** User needed to copy ROI coordinates for use in other code
- **Implementation:** get_bounds(), get_roi_code(), print_roi_code() methods
- **Committed in:** 6b50385

---

**Total deviations:** 2 user-requested enhancements
**Impact on plan:** Enhanced functionality beyond original scope, no issues

## Issues Encountered

None

## Next Phase Readiness

- ROIWidget core complete with both selection tools
- Ready for 14-02: Multi-class ROI labeling and GroundTruth export
- Coordinate extraction enables integration with existing workflows

---
*Phase: 14-jupyter-roi-widget*
*Completed: 2026-01-21*
