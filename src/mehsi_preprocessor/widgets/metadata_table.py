"""QTableWidget wrapper for displaying a pandas DataFrame."""

from __future__ import annotations

from typing import Optional

import pandas as pd
from PyQt6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget


class MetadataTable(QWidget):
    """Read-only table that displays a ``pandas.DataFrame``."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._table)

    def set_dataframe(self, df: pd.DataFrame) -> None:
        """Populate the table from *df*."""
        self._table.clear()
        self._table.setRowCount(len(df))
        self._table.setColumnCount(len(df.columns))
        self._table.setHorizontalHeaderLabels([str(c) for c in df.columns])

        for r in range(len(df)):
            for c in range(len(df.columns)):
                val = df.iat[r, c]
                item = QTableWidgetItem(str(val))
                self._table.setItem(r, c, item)

        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

    def clear(self) -> None:
        self._table.clear()
        self._table.setRowCount(0)
        self._table.setColumnCount(0)
