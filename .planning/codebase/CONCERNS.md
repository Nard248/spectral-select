# Codebase Concerns

**Analysis Date:** 2026-01-19

## Tech Debt

**Hardcoded Windows Paths (CRITICAL):**
- Issue: Absolute Windows paths hardcoded throughout codebase
- Files:
  - `wavelength_analysis/WavelengthSelectionFinal.py` line 29: `base_dir = Path(r"C:\Users\meloy\PycharmProjects\Capstone")`
  - `wavelength_analysis/wavelengthSelectionV2-2-PCA.py` line 34: Same pattern
  - `wavelength_analysis/robustness_analysis.py`: `MASK_PATH = Path(r"C:\Users\meloy\Downloads\Mask_Manual.png")`
  - `wavelength_analysis/test_v2_pipeline.py`, `wavelength_analysis/test_roi_overlay_v2.py`: Hardcoded test paths
- Why: Rapid development without portability consideration
- Impact: Code completely breaks on macOS/Linux, different users, or different directories
- Fix approach: Use environment variables, config files, or relative paths

**Multiple Competing Code Versions (HIGH):**
- Issue: 5+ versions of wavelength selection with overlapping functionality
- Files:
  - `wavelength_analysis/wavelengthSelectionV2.py` (~1,143 lines)
  - `wavelength_analysis/wavelengthselectionV2-2.py` (~1,474 lines)
  - `wavelength_analysis/wavelengthSelectionV2-2-PCA.py` (~1,608 lines)
  - `wavelength_analysis/wavelengthSelectionV2SeparateObjectAnalysis.py`
  - `wavelength_analysis/wavelengthSelectionV2SeparateObjectAnalysis_old.py` (~764 lines)
  - `wavelength_analysis/WavelengthSelectionFinal.py` (~1,793 lines)
- Why: Incremental development without consolidation
- Impact: Bug fixes must be applied to multiple files, versions diverge
- Fix approach: Consolidate into single implementation with configuration options

**Print Statement Logging (MEDIUM):**
- Issue: 1,550+ print statements instead of proper logging
- Files: Throughout `wavelength_analysis/` directory
- Why: Quick debugging during development
- Impact: Cannot control verbosity, no log levels, output clutters production
- Fix approach: Replace with Python logging module, configure log levels

**sys.path Manipulation (MEDIUM):**
- Issue: 30+ instances of `sys.path.append()` and `sys.path.insert()`
- Files: `wavelength_analysis/core/analyzer.py` line 23, many others
- Why: Workaround for import issues without proper package structure
- Impact: Fragile imports, breaks when run from different directories
- Fix approach: Proper package installation with setup.py/pyproject.toml

## Known Bugs

**Bare Exception Handlers:**
- Symptoms: Errors silently swallowed, metrics return 0/None unexpectedly
- Files:
  - `wavelength_analysis/enhanced_viz_module.py` lines 363, 368, 373, 526
  - `wavelength_analysis/wavelengthselectionV2.py` line 447
  - `wavelength_analysis/wavelengthselectionV2-2.py` line 456
  - `wavelength_analysis/wavelengthSelectionV2-2-PCA.py` lines 1429-1440
  - `wavelength_analysis/object_wise_metrics.py` lines 187, 212
  - `scripts/utils/masking_tool.py`: Multiple instances
- Trigger: Any exception in metric calculation or file I/O
- Workaround: None - errors are hidden
- Root cause: Defensive coding without proper exception handling
- Fix: Replace bare `except:` with specific exception types and logging

**Deprecated Function Still Used:**
- Symptoms: Deprecation warning during execution
- File: `wavelength_analysis/concatenation_clustering.py` lines 224-232
- Trigger: Calling deprecated function instead of `concatenate_hyperspectral_data_improved()`
- Workaround: Warning only, function still works
- Fix: Update all call sites to use improved version

## Security Considerations

**No Significant Security Risks:**
- Risk: N/A - Local scientific computing tool, no network exposure
- Current mitigation: N/A
- Recommendations: N/A

**File Path Injection (LOW):**
- Risk: User-provided paths could access unintended files
- Files: Data loading functions accepting arbitrary paths
- Current mitigation: None (not internet-facing)
- Recommendations: Validate paths are within expected directories

## Performance Bottlenecks

**Large Monolithic Files:**
- Problem: Files with 1,000+ lines are slow to parse and difficult to maintain
- Files:
  - `MCR-Analysis/mcr_als.py`: 2,526 lines
  - `wavelength_analysis/WavelengthSelectionFinal.py`: 1,793 lines
  - `wavelength_analysis/wavelengthSelectionV2-2-PCA.py`: 1,608 lines
  - `scripts/models/clustering.py`: 1,327 lines
  - `scripts/utils/masking_tool.py`: 1,162 lines
- Cause: Incremental development without refactoring
- Improvement path: Split into smaller, focused modules

**No GPU Memory Management:**
- Problem: PyTorch models may not release GPU memory between runs
- Files: `wavelength_analysis/core/analyzer.py`, `scripts/models/training.py`
- Cause: No explicit `torch.cuda.empty_cache()` or model cleanup
- Improvement path: Add explicit memory cleanup after model operations

## Fragile Areas

**Analysis Orchestrator:**
- File: `wavelength_analysis/core/analyzer.py` (754 lines)
- Why fragile: Central coordinator with many dependencies, complex state
- Common failures: Import errors from path manipulation, missing data files
- Safe modification: Test thoroughly, check all code paths
- Test coverage: Limited - tests use hardcoded paths that don't work

**Configuration Presets:**
- File: `wavelength_analysis/core/config.py`
- Why fragile: Presets have hardcoded paths and sample-specific settings
- Common failures: Path not found errors on different machines
- Safe modification: Update all presets together, test with multiple samples
- Test coverage: None detected

## Scaling Limits

**Memory Usage:**
- Current capacity: Single machine with sufficient RAM for 4D hyperspectral arrays
- Limit: Large hyperspectral datasets may exceed available RAM
- Symptoms at limit: MemoryError, system slowdown
- Scaling path: Implement chunked processing, use memory-mapped arrays

## Dependencies at Risk

**requirements.txt Encoding:**
- Risk: File has encoding issues (UTF-16LE) that may cause parsing problems
- Impact: pip install may fail with encoding errors
- Migration plan: Re-save as UTF-8

**No Dependency Pinning:**
- Risk: Breaking changes from dependency updates
- Impact: Code may break with new numpy, torch, or sklearn versions
- Migration plan: Create requirements.txt with exact versions (`==`)

## Missing Critical Features

**Proper Package Structure:**
- Problem: No setup.py or pyproject.toml for installation
- Current workaround: sys.path manipulation
- Blocks: Cannot pip install, cannot distribute
- Implementation complexity: Medium - create pyproject.toml

**Configuration Management:**
- Problem: No environment-based configuration
- Current workaround: Hardcoded paths, manual editing
- Blocks: Cannot run on different machines without code changes
- Implementation complexity: Low - add .env support with python-dotenv

## Test Coverage Gaps

**Integration Tests:**
- What's not tested: Full pipeline from data loading to visualization
- Risk: End-to-end failures not caught
- Priority: High
- Difficulty: Tests have hardcoded paths, need portable test data

**Error Handling:**
- What's not tested: Exception paths, error recovery
- Risk: Errors cause silent failures or crashes
- Priority: High
- Difficulty: Need to mock failure scenarios

**Cross-Platform:**
- What's not tested: macOS/Linux compatibility
- Risk: Windows-only code won't work on other platforms
- Priority: Critical (current environment is macOS)
- Difficulty: Requires path abstraction throughout codebase

---

## Summary Statistics

| Category | Count | Severity |
|----------|-------|----------|
| Hardcoded Windows paths | 10+ files | CRITICAL |
| Bare `except:` blocks | 20+ | HIGH |
| Duplicate code versions | 5+ variants | HIGH |
| Print statements | 1,550+ | MEDIUM |
| Files >1,000 lines | 10 | MEDIUM |
| sys.path manipulation | 30+ | MEDIUM |
| Test files (scattered) | 8 | MEDIUM |

## Recommended Priority Fixes

1. **CRITICAL**: Replace hardcoded Windows paths with config/env vars
2. **HIGH**: Replace bare `except:` with proper exception handling
3. **HIGH**: Consolidate wavelength selection variants into single implementation
4. **MEDIUM**: Migrate print statements to Python logging
5. **MEDIUM**: Refactor files >1,000 lines into smaller modules
6. **MEDIUM**: Create proper package structure with pyproject.toml
7. **LOW**: Organize tests into standard `tests/` directory

---

*Concerns audit: 2026-01-19*
*Update as issues are fixed or new ones discovered*
