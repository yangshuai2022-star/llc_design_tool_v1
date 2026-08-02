"""Clickable block-level LLC topology view."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QPen
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)


class _ClickableBlock(QGraphicsRectItem):
    def __init__(self, key: str, label: str, x: float, y: float, width: float, height: float):
        super().__init__(x, y, width, height)
        self.key = key
        self.setBrush(QBrush(Qt.GlobalColor.white))
        self.setPen(QPen(Qt.GlobalColor.darkGray, 1.2))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        text = QGraphicsSimpleTextItem(label, self)
        bounds = text.boundingRect()
        text.setPos(x + (width - bounds.width()) / 2.0, y + (height - bounds.height()) / 2.0)


class TopologyView(QGraphicsView):
    component_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        scene = QGraphicsScene(self)
        self.setScene(scene)
        self.setMinimumHeight(210)
        self.setSceneRect(0, 0, 850, 190)
        blocks = [
            ("bus", "DC Bus", 20, 70, 90, 50),
            ("bridge", "Full Bridge", 140, 70, 100, 50),
            ("lr", "Lr", 275, 70, 65, 50),
            ("cr", "Cr", 375, 70, 65, 50),
            ("transformer", "Transformer", 475, 70, 110, 50),
            ("sr", "Full Bridge SR", 620, 70, 110, 50),
            ("output", "Co / Load", 760, 70, 80, 50),
        ]
        self.blocks: dict[str, _ClickableBlock] = {}
        for key, label, x, y, w, h in blocks:
            block = _ClickableBlock(key, label, x, y, w, h)
            scene.addItem(block)
            self.blocks[key] = block
        for left, right in zip(blocks[:-1], blocks[1:]):
            x0 = left[2] + left[4]
            x1 = right[2]
            y = left[3] + left[5] / 2.0
            scene.addLine(x0, y, x1, y, QPen(Qt.GlobalColor.black, 1.5))
        node_labels = [
            ("Vab", 245, 48), ("Ir", 340, 48), ("VCr", 440, 48),
            ("Vp/Ip", 570, 48), ("Vs/Is", 710, 48), ("Vo/ICo", 790, 135),
        ]
        for label, x, y in node_labels:
            scene.addSimpleText(label).setPos(x, y)

    def mousePressEvent(self, event):  # noqa: N802 - Qt API
        item = self.itemAt(event.position().toPoint())
        while item is not None and not isinstance(item, _ClickableBlock):
            item = item.parentItem()
        if isinstance(item, _ClickableBlock):
            self.component_selected.emit(item.key)
        super().mousePressEvent(event)
