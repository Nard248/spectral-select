# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-19)

**Core value:** Clean, stable, reproducible wavelength selection analysis that anyone can `from spectral_select import Analyzer` and use immediately.
**Current focus:** Phase 8 — Testing & Validation (Ready to plan)

## Current Position

Phase: 7 of 8 (Notebook Migration)
Plan: 1 of 1 in current phase
Status: Phase complete
Last activity: 2026-01-19 — Completed 07-01-PLAN.md

Progress: █████████████░ 85%

## Performance Metrics

**Velocity:**
- Total plans completed: 16
- Average duration: 5.0 min
- Total execution time: 1.4 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Package Structure | 2 | 6 min | 3 min |
| 2. Config System | 2 | 6 min | 3 min |
| 3. Core Data Types | 2 | 9 min | 4.5 min |
| 4. Analysis Engine | 4 | 26 min | 6.5 min |
| 5. Visualization Module | 3 | 16 min | 5.3 min |
| 6. Ground Truth Validation | 2 | 13 min | 6.5 min |
| 7. Notebook Migration | 1 | 4 min | 4 min |

**Recent Trend:**
- Last 5 plans: 05-03 (7 min), 06-01 (8 min), 06-02 (5 min), 07-01 (4 min)
- Trend: Phase 7 complete, example notebooks created

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- **01-01:** Removed readme field from pyproject.toml (UTF-16 encoding issue)
- **01-01:** Used flat layout (spectral_select/ at root) instead of src/ layout
- **01-02:** Export classes at package root for clean imports
- **01-02:** Placeholder classes with phase references for future implementation
- **02-01:** TYPE_CHECKING guard for forward protocol references
- **02-01:** Protocol-based component interfaces with @runtime_checkable
- **02-01:** Dual registration pattern (string IDs + direct class/callable)
- **02-02:** Use yaml.safe_load/safe_dump for security
- **02-02:** Warn on unknown config keys (forward compatibility)
- **02-02:** Auto-create parent directories when saving configs
- **03-01:** Exclude large arrays from SpectraData.to_dict() for lightweight serialization
- **03-01:** Generate placeholder emission wavelengths when loading existing pkl format
- **03-01:** Allow empty excitations dict for SpectraData initialization
- **03-02:** JSON serialization for results (human-readable, portable)
- **03-02:** Factory method from_bands() for computing metrics from selection list
- **03-02:** Validate sequential ranks in WavelengthResult.__post_init__
- **04-01:** Private attributes with public properties for Analyzer encapsulation
- **04-01:** Device fallback chain: configured → availability check → cpu
- **04-01:** fit() returns self for sklearn-style method chaining
- **04-02:** torch.load with map_location for device-agnostic model loading
- **04-02:** Train with 3000 epochs, lr=0.001 defaults (matching original)
- **04-02:** Filter patches by >50% mask validity for baseline extraction
- **04-03:** Three dimension selection methods: variance, activation, pca
- **04-03:** Three perturbation methods: percentile, standard_deviation, absolute_range
- **04-03:** Use 1e-10 epsilon for division by zero protection
- **04-04:** MMR uses cosine similarity on flattened spectral profiles
- **04-04:** Min-distance constraint applies only within same excitation
- **04-04:** TIFF layers saved as 16-bit normalized images
- **05-01:** HUSL palette (12 colors) for perceptually uniform visualization
- **05-01:** Factory methods auto-generate output_dir from sample_name
- **05-01:** TYPE_CHECKING import pattern for circular import prevention
- **05-02:** Log10 + 1e-10 for heatmap values (handles zeros gracefully)
- **05-02:** Auto log scale in ranking when max/min > 100x
- **05-02:** Size encoding inversely proportional to rank in scatter plots
- **05-03:** Row-wise normalization for confusion matrix (per true class)
- **05-03:** Red/green colormap for accuracy heatmap (intuitive incorrect/correct)
- **05-03:** 3-panel ROI overlay: clustering result, ROI boxes, accuracy chart
- **05-03:** Graceful degradation in plot_all: continue if one plot fails
- **06-01:** Validator.score() returns ARI as primary metric (sklearn convention)
- **06-01:** Background pixels use -1 convention throughout
- **06-01:** PNG loader uses 30px tolerance for background, configurable for classes
- **06-02:** Store flattened ground_truth in Validator.fit() for later retrieval
- **06-02:** Report format uses Markdown tables for GitHub/Jupyter rendering
- **06-02:** Visualizer stores validation data in private attributes

### Deferred Issues

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-01-19
Stopped at: Completed 07-01-PLAN.md (Phase 7 complete)
Resume file: None
