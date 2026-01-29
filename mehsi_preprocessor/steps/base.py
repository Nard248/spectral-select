"""Base class for all wizard step widgets."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QWidget

if TYPE_CHECKING:
    from mehsi_preprocessor.state import PipelineState


class AbstractStepWidget(QWidget):
    """Base for every wizard step panel.

    Subclasses must implement:
        * :meth:`step_index` – the 1-based step number
        * :meth:`title` – human-readable step name
        * :meth:`on_enter` – called when the step becomes active
        * :meth:`on_leave` – called when leaving the step (return *False* to block)
    """

    def __init__(self, state: PipelineState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def step_index(self) -> int:
        """1-based step number."""
        ...

    @property
    @abstractmethod
    def title(self) -> str:
        """Short label shown in the sidebar."""
        ...

    @abstractmethod
    def on_enter(self) -> None:
        """Called every time this step is shown.

        Use this to refresh displays based on the current pipeline state.
        """
        ...

    def on_leave(self) -> bool:
        """Called when navigating away.  Return *False* to prevent leaving."""
        return True
