---
phase: 13-masking-gui-tool
plan: 01
subsystem: ui
tags: [tkinter, matplotlib, viewer, gui, visualization]

# Dependency graph
requires:
  - phase: 12-data-pipeline-improvements
    provides: SpectraData.from_pickle(), DataLoader for raw files
provides:
  - ViewerApp class for interactive hyperspectral viewing
  - launch_viewer() convenience function
  - create_rgb_image() helper for cube visualization
  - detect_cube_format() helper for dimension detection
  - get_pixel_value() helper for cursor tracking
affects: [14-jupyter-roi-widget, masking-tools]

# Tech tracking
tech-stack:
  added: [tkinter, matplotlib.backends.backend_tkagg]
  patterns: [lazy-import-circular-avoidance, property-encapsulation]

key-files:
  created: [spectral_select/viewer.py]
  modified: [spectral_select/__init__.py]

key-decisions:
  - "Use tkinter + matplotlib FigureCanvasTkAgg (stdlib + existing dep)"
  - "Lazy SpectraData import in viewer to avoid circular imports"
  - "Graceful ImageJ failure with info dialog for raw file loading"
  - "Private attributes with public properties for state encapsulation"

patterns-established:
  - "GUI state management with tkinter Variables (BooleanVar, StringVar)"
  - "Matplotlib canvas embedding pattern for tkinter"
  - "Progressive file type detection (pkl vs im3)"

issues-created: []

# Metrics
duration: 6min
completed: 2026-01-21
---

# Phase 13 Plan 01: ME-HSI Viewer Core Summary

**Tkinter-based ME-HSI Viewer with professional window layout, SpectraData loading, and matplotlib canvas visualization**

## Performance

- **Duration:** 6 min
- **Started:** 2026-01-21T05:52:44Z
- **Completed:** 2026-01-21T05:59:39Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments

- ViewerApp class with professional 1200x800 window layout
- Data loading support for both pkl (SpectraData) and raw im3 files
- Graceful ImageJ unavailability handling with helpful info dialog
- Control panel with Data, Excitation, Display, and Tools sections
- Matplotlib canvas with NavigationToolbar (zoom, pan, home, save)
- Info bar showing cursor position, pixel value, and excitation info
- Display controls: auto-contrast, min/max, RGB method selection
- Keyboard shortcuts: Ctrl+O (open), R (reset view), Escape (cancel)
- Helper functions exported at package level for testability

## Task Commits

Each task was committed atomically:

1. **Task 1: Create main window layout** - `96b5244` (feat)
2. **Task 2: Implement data loading** - `fdafaaf` (feat)
3. **Task 3: Add display controls and exports** - `12325d7` (feat)

**Plan metadata:** (pending docs commit)

## Files Created/Modified

- `spectral_select/viewer.py` - New ViewerApp class with all viewer functionality
- `spectral_select/__init__.py` - Added ViewerApp, launch_viewer, create_rgb_image, detect_cube_format exports

## Decisions Made

- Used tkinter (stdlib) + matplotlib FigureCanvasTkAgg for GUI - matches existing masking_tool.py pattern
- Lazy import of SpectraData in viewer methods to avoid circular imports at module level
- Graceful ImageJ failure - shows info dialog with install instructions instead of error
- Private attributes (_spectra_data, _current_excitation) with public properties for encapsulation
- Helper functions (create_rgb_image, detect_cube_format, get_pixel_value) are module-level for testability

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## Next Phase Readiness

- ViewerApp core is complete and exported from package
- Ready for 13-02: Basic masking tools (polygon, rectangle drawing)
- Pattern established for tkinter + matplotlib canvas integration
- Info bar infrastructure ready for mask status display

---
*Phase: 13-masking-gui-tool*
*Completed: 2026-01-21*
