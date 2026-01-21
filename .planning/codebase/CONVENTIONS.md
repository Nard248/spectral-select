# Coding Conventions

**Analysis Date:** 2026-01-19

## Naming Patterns

**Files:**
- snake_case for organized modules: `analyzer.py`, `config.py`, `hyperspectral_loader.py`
- PascalCase in legacy code: `WavelengthSelectionFinal.py`, `DIAGNOSTIC_ROI_ISSUE.py`
- Test files: `test_*.py` alongside source files
- Entry points: `run_*.py`

**Functions:**
- snake_case: `load_data_and_model()`, `apply_spectral_cutoff()`, `extract_encoded_features()`
- No special prefix for async functions
- Handler pattern: `handle_*` not commonly used

**Variables:**
- snake_case: `excitation_wavelengths`, `ground_truth`, `output_dir`
- Constants: UPPER_SNAKE_CASE limited (`ROI_REGIONS`)
- No underscore prefix for private members

**Types:**
- PascalCase for classes: `WavelengthAnalyzer`, `AnalysisConfig`, `MaskedHyperspectralDataset`
- No I prefix for interfaces
- Domain suffixes: `*Analyzer`, `*Dataset`, `*Processor`, `*Loader`, `*Visualizer`

## Code Style

**Formatting:**
- 4 spaces per indentation level (Python standard)
- No explicit formatter configured (no .prettierrc, black.toml)
- Mixed quote styles (single and double)
- No semicolons (Python standard)

**Linting:**
- No linting configuration detected
- No .flake8, .pylintrc, pyproject.toml with tools
- IDE (PyCharm) may provide default checks

## Import Organization

**Order (observed):**
1. Standard library imports (os, sys, pathlib, typing)
2. Third-party packages (numpy, pandas, torch, sklearn)
3. Local imports (from .config, from wavelength_analysis)
4. No explicit type-only imports

**Grouping:**
- Blank lines between groups
- Alphabetical sorting not enforced
- Star imports occasionally used

**Path Manipulation (Technical Debt):**
- 30+ instances of `sys.path.append()` and `sys.path.insert()`
- Example: `wavelength_analysis/core/analyzer.py` line 23

## Error Handling

**Patterns:**
- Try/except at function boundaries
- Many bare `except:` clauses (technical debt)
- Warnings module for non-fatal issues

**Error Types:**
- Custom exceptions not widely used
- Standard exceptions (ValueError, FileNotFoundError)
- Silent error swallowing with `pass` (technical debt)

## Logging

**Framework:**
- Mixed: Python logging module in some files
- 1,550+ print statements throughout (technical debt)

**Patterns:**
- `logging.basicConfig()` with INFO level
- Logger per module: `logger = logging.getLogger(__name__)`
- Check emoji indicators in tests (✓, ✗)

## Comments

**When to Comment:**
- Module docstrings at file top
- Class docstrings after class definition
- Function docstrings in Google/NumPy style

**JSDoc/TSDoc (Python Docstrings):**
- Google style with Args, Returns sections
- Example from `wavelength_analysis/core/analyzer.py`:
```python
"""
Initialize the wavelength analyzer.

Args:
    config: Analysis configuration object
"""
```

**TODO Comments:**
- Pattern: `# TODO: description`
- FIXME, HACK also present
- No issue linking convention

## Function Design

**Size:**
- Large functions common (>50 lines)
- Giant monolithic scripts (1,000+ lines) - technical debt

**Parameters:**
- Optional parameters with `Optional[T]` type hints
- Dataclasses for configuration (`AnalysisConfig`)
- Destructuring in parameter list not used

**Return Values:**
- Explicit return statements
- Tuple returns for multiple values
- Dict returns for complex results

## Module Design

**Exports:**
- `__init__.py` with explicit exports
- No barrel file pattern
- Direct imports common

**Package Structure:**
- `wavelength_analysis/core/` for abstractions
- `scripts/models/` for ML code
- Relative imports within packages (`.config`, `.analyzer`)

## Type Hints

**Usage:**
- Comprehensive type hints in core modules
- `from typing import List, Dict, Any, Optional, Tuple`
- Return type annotations present

**Examples from codebase:**
- `data_path: Optional[str] = None`
- `-> Dict:`
- `-> Tuple[np.ndarray, List[float]]`

**Dataclass Pattern:**
```python
@dataclass
class AnalysisConfig:
    sample_name: str = "Lime"
    dimension_selection_method: str = "variance"
```

---

*Convention analysis: 2026-01-19*
*Update when patterns change*
