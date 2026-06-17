"""Tests for the train/select PipelineState fields and the headless worker job."""
from mehsi_preprocessor.state import PipelineState, STEP_TRAIN, STEP_SELECT


def test_state_has_train_select_fields_and_invalidation():
    s = PipelineState()
    assert s.analyzer is None
    assert s.training_losses == []
    assert s.selection_result is None

    # Leaving the Train step invalidates the (downstream) selection, keeps the model.
    s.analyzer = object()
    s.selection_result = object()
    s.invalidate_from(STEP_TRAIN)
    assert s.selection_result is None
    assert s.analyzer is not None

    # Re-doing an earlier step (ROI = 7) invalidates both model and selection.
    s.selection_result = object()
    s.invalidate_from(7)
    assert s.analyzer is None
    assert s.selection_result is None


def test_step_constants():
    assert STEP_TRAIN == 9
    assert STEP_SELECT == 10
