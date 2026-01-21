# Testing Patterns

**Analysis Date:** 2026-01-19

## Test Framework

**Runner:**
- pytest (convention-based, no pytest.ini)
- No conftest.py detected

**Assertion Library:**
- Standard assert statements
- Manual verification with logging

**Run Commands:**
```bash
pytest wavelength_analysis/test_*.py        # Run all tests
pytest wavelength_analysis/test_v2_pipeline.py  # Single file
# No coverage configuration detected
```

## Test File Organization

**Location:**
- Collocated with source: `wavelength_analysis/test_*.py`
- No separate `tests/` directory

**Naming:**
- Pattern: `test_*.py` (pytest convention)
- No integration/e2e suffix distinction

**Structure:**
```
wavelength_analysis/
├── test_v2_pipeline.py
├── test_object_wise_analysis.py
├── test_knn_roi.py
├── test_roi_overlay_v2.py
├── test_simple_object_wise.py
├── test_visualization_integration.py
├── test_object_wise_with_synthetic.py
└── archive/
    └── test_ground_truth_validation.py
```

## Test Structure

**Suite Organization:**
```python
def test_object_segmentation():
    """Test the object segmentation module."""
    logger.info("\n" + "="*60)
    logger.info("Testing Object Segmentation Module")
    logger.info("="*60)

    try:
        from object_segmentation import ObjectSegmentation
        # ... setup code ...
        logger.info(f"✓ Successfully segmented {len(objects)} objects")
        return True
    except Exception as e:
        logger.error(f"✗ Object segmentation test failed: {e}")
        return False
```

**Patterns:**
- Function-based tests (no classes)
- Try/except with boolean return values
- Logging for progress indication
- Emoji indicators (✓, ✗) for pass/fail

## Mocking

**Framework:**
- Not detected (no unittest.mock usage found)

**Patterns:**
- Real data used in tests
- Synthetic data generation for isolated testing

**What to Mock:**
- Not currently practiced

**What NOT to Mock:**
- Tests use actual file I/O and data processing

## Fixtures and Factories

**Test Data:**
```python
# Synthetic data generation in test files
def test_object_metrics():
    # Create synthetic ground truth and predictions
    ground_truth = np.array([...])
    predictions = np.array([...])
```

**Location:**
- Inline in test files
- No shared fixtures directory
- No pytest fixtures defined

## Coverage

**Requirements:**
- No coverage target enforced
- No coverage configuration

**Configuration:**
- Not configured

**View Coverage:**
```bash
# Not currently set up
# pytest --cov=wavelength_analysis would work if configured
```

## Test Types

**Unit Tests:**
- `test_object_segmentation()` - Single module testing
- `test_object_metrics()` - Metrics calculation testing

**Integration Tests:**
- `test_full_pipeline_integration()` - Multi-module testing
- `test_visualization_integration.py` - Visualization pipeline

**E2E Tests:**
- Not formally separated
- Pipeline tests serve as E2E

## Common Patterns

**Async Testing:**
- Not applicable (synchronous code)

**Error Testing:**
```python
try:
    # Test code that might fail
    result = some_function()
    logger.info(f"✓ Test passed")
    return True
except Exception as e:
    logger.error(f"✗ Test failed: {e}")
    return False
```

**Data Shape Assertions:**
```python
assert len(objects) > 0, "Should segment at least one object"
assert predictions.shape == ground_truth.shape
```

**Snapshot Testing:**
- Not used

## Test Files Found

| File | Purpose |
|------|---------|
| `test_v2_pipeline.py` | Pipeline component tests |
| `test_object_wise_analysis.py` | Object segmentation and metrics |
| `test_knn_roi.py` | KNN-based ROI analysis |
| `test_roi_overlay_v2.py` | ROI overlay visualization |
| `test_simple_object_wise.py` | Simplified object analysis |
| `test_visualization_integration.py` | Visualization methods |
| `test_object_wise_with_synthetic.py` | Synthetic data testing |
| `archive/test_ground_truth_validation.py` | Legacy validation tests |

## Issues & Technical Debt

**Test Organization:**
- Tests scattered in main module, not in `tests/` directory
- Archived tests may be outdated

**Test Infrastructure:**
- No pytest.ini or conftest.py
- No coverage configuration
- No CI/CD integration visible

**Test Reliability:**
- Hardcoded paths in some test files
- Boolean return values instead of assertions
- Manual logging instead of pytest reporting

---

*Testing analysis: 2026-01-19*
*Update when test patterns change*
