"""Interactive vector control block diagram used by LLC and PFC workspaces.

V7.1.3 adds the interaction model needed by the design workbench:

* semantic block selection for parameter/Bode linkage;
* vector zoom, pan, fit-to-window and 100 % view;
* selected-path highlighting so the active signal path is obvious;
* reusable diagrams for overview, focused detail and full-screen inspection.

The diagram deliberately stays lightweight: all blocks, labels and connections are
QGraphicsItems, so the view remains sharp at any zoom level and no bitmap scaling is
involved.
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class BlockSpec:
    key: str
    title: str
    subtitle: str = ""
    x: float = 0.0
    y: float = 0.0
    width: float = 150.0
    height: float = 72.0
    role: str = "normal"  # normal/controller/plant/sense/modulator/aux


@dataclass(frozen=True)
class ConnectionSpec:
    source: str
    target: str
    label: str = ""
    feedback: bool = False


@dataclass
class _ConnectionGraphics:
    spec: ConnectionSpec
    lines: list
    arrow: object
    label: QGraphicsSimpleTextItem | None
    base_color: QColor


class _SignalProxy(QObject):
    selected = Signal(str)


_ROLE_COLORS = {
    "normal": (QColor("#f8fafc"), QColor("#64748b")),
    "controller": (QColor("#eef4ff"), QColor("#175cd3")),
    "plant": (QColor("#ecfdf3"), QColor("#0e9384")),
    "sense": (QColor("#fff7ed"), QColor("#ea580c")),
    "modulator": (QColor("#f5f3ff"), QColor("#7c3aed")),
    "aux": (QColor("#fdf2fa"), QColor("#c11574")),
}


class _BlockItem(QGraphicsRectItem):
    def __init__(self, spec: BlockSpec, proxy: _SignalProxy):
        super().__init__(0.0, 0.0, spec.width, spec.height)
        self.spec = spec
        self.proxy = proxy
        self.setPos(spec.x, spec.y)
        fill, edge = _ROLE_COLORS.get(spec.role, _ROLE_COLORS["normal"])
        self.normal_brush = QBrush(fill)
        self.normal_pen = QPen(edge, 2.0)
        self.selected_pen = QPen(QColor("#111827"), 3.2)
        self.setBrush(self.normal_brush)
        self.setPen(self.normal_pen)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"{spec.title}\n{spec.subtitle}" if spec.subtitle else spec.title)

        title = QGraphicsSimpleTextItem(spec.title, self)
        font = QFont()
        font.setPointSize(10 if spec.height < 58.0 else 11)
        font.setBold(True)
        title.setFont(font)
        title_rect = title.boundingRect()
        title_y = 8.0 if spec.height < 58.0 else 13.0
        title.setPos((spec.width - title_rect.width()) / 2.0, title_y)

        if spec.subtitle:
            subtitle = QGraphicsSimpleTextItem(spec.subtitle, self)
            font2 = QFont()
            font2.setPointSize(8 if spec.height < 58.0 else 9)
            subtitle.setFont(font2)
            subtitle.setBrush(QBrush(QColor("#475467")))
            rect = subtitle.boundingRect()
            subtitle_y = max(title_y + title_rect.height() + 3.0, spec.height - rect.height() - 7.0)
            subtitle.setPos((spec.width - rect.width()) / 2.0, subtitle_y)

    def mousePressEvent(self, event):
        self.proxy.selected.emit(self.spec.key)
        super().mousePressEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self.setPen(self.selected_pen if bool(value) else self.normal_pen)
        return super().itemChange(change, value)


class _DiagramGraphicsView(QGraphicsView):
    """QGraphicsView with CAD-like wheel zoom and double-click fit."""

    def __init__(self, scene: QGraphicsScene, owner: "ControlBlockDiagram"):
        super().__init__(scene)
        self._owner = owner
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        self._owner.zoom_by(1.16 if delta > 0 else 1.0 / 1.16)
        event.accept()

    def mouseDoubleClickEvent(self, event):
        # Double-clicking blank space is a quick "fit" command.  Double-clicking
        # a block is left to the block item so normal selection still works.
        if self.itemAt(event.position().toPoint()) is None:
            self._owner.fit_to_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class ControlBlockDiagram(QWidget):
    """Reusable clickable vector control diagram.

    ``block_selected`` emits a stable semantic block key. ``select_block`` is
    used by parameter tabs or Bode selectors to drive the diagram in reverse.
    The view supports wheel zoom, hand panning, fit-to-window and selected-path
    highlighting.
    """

    block_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.view = _DiagramGraphicsView(self.scene, self)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.view.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumHeight(150)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.view.setStyleSheet(
            "QGraphicsView {background:#ffffff;border:1px solid #d0d5dd;border-radius:6px;}"
        )
        self._proxy = _SignalProxy()
        self._proxy.selected.connect(self._on_selected)
        self._items: dict[str, _BlockItem] = {}
        self._connections: list[ConnectionSpec] = []
        self._connection_graphics: list[_ConnectionGraphics] = []
        self._blocks: list[BlockSpec] = []
        self._auto_fit = True
        self._selected_key: str | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)

    @property
    def block_specs(self) -> tuple[BlockSpec, ...]:
        return tuple(self._blocks)

    @property
    def connection_specs(self) -> tuple[ConnectionSpec, ...]:
        return tuple(self._connections)

    @property
    def selected_key(self) -> str | None:
        return self._selected_key

    def set_diagram(self, blocks: list[BlockSpec], connections: list[ConnectionSpec]) -> None:
        self.scene.clear()
        self._items.clear()
        self._connection_graphics.clear()
        self._blocks = list(blocks)
        self._connections = list(connections)
        self._selected_key = None
        for spec in blocks:
            item = _BlockItem(spec, self._proxy)
            self.scene.addItem(item)
            self._items[spec.key] = item
        for connection in connections:
            self._draw_connection(connection)
        rect = self.scene.itemsBoundingRect().adjusted(-18, -16, 18, 16)
        self.scene.setSceneRect(rect)
        self.fit_to_view()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._auto_fit and not self.scene.sceneRect().isEmpty():
            self._fit_now()

    def _fit_now(self) -> None:
        rect = self.scene.sceneRect()
        if not rect.isEmpty() and self.view.viewport().width() > 4 and self.view.viewport().height() > 4:
            self.view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def fit_to_view(self) -> None:
        self._auto_fit = True
        self._fit_now()

    def actual_size(self) -> None:
        self._auto_fit = False
        self.view.resetTransform()
        if not self.scene.sceneRect().isEmpty():
            self.view.centerOn(self.scene.sceneRect().center())

    def zoom_by(self, factor: float) -> None:
        if factor <= 0.0:
            return
        current = abs(self.view.transform().m11())
        target = current * factor
        if target < 0.35 or target > 5.5:
            return
        self._auto_fit = False
        self.view.scale(factor, factor)

    def zoom_in(self) -> None:
        self.zoom_by(1.18)

    def zoom_out(self) -> None:
        self.zoom_by(1.0 / 1.18)

    def focus_block(self, key: str) -> None:
        item = self._items.get(key)
        if item is None:
            return
        rect = item.sceneBoundingRect().adjusted(-85.0, -65.0, 85.0, 65.0)
        self._auto_fit = False
        self.view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def _on_selected(self, key: str) -> None:
        self.select_block(key, emit=False)
        self.block_selected.emit(key)

    def select_block(self, key: str, *, emit: bool = False) -> None:
        if key not in self._items:
            return
        self._selected_key = key
        for item_key, item in self._items.items():
            item.setSelected(item_key == key)
        self._apply_signal_path_highlight(key)
        if emit:
            self.block_selected.emit(key)

    def clear_selection(self) -> None:
        self._selected_key = None
        for item in self._items.values():
            item.setSelected(False)
            item.setOpacity(1.0)
        for graphics in self._connection_graphics:
            self._style_connection(graphics, highlighted=False, dimmed=False)

    def has_block(self, key: str) -> bool:
        return key in self._items

    def _apply_signal_path_highlight(self, key: str) -> None:
        neighbours = {key}
        for connection in self._connections:
            if connection.source == key:
                neighbours.add(connection.target)
            elif connection.target == key:
                neighbours.add(connection.source)

        for item_key, item in self._items.items():
            if item_key == key:
                item.setOpacity(1.0)
            elif item_key in neighbours:
                item.setOpacity(0.92)
            else:
                item.setOpacity(0.42)

        for graphics in self._connection_graphics:
            related = graphics.spec.source == key or graphics.spec.target == key
            self._style_connection(graphics, highlighted=related, dimmed=not related)

    @staticmethod
    def _style_connection(
        graphics: _ConnectionGraphics,
        *,
        highlighted: bool,
        dimmed: bool,
    ) -> None:
        width = 3.0 if highlighted else 1.8
        pen = QPen(graphics.base_color, width)
        pen.setCosmetic(True)
        opacity = 1.0 if highlighted else (0.22 if dimmed else 1.0)
        for line in graphics.lines:
            line.setPen(pen)
            line.setOpacity(opacity)
        graphics.arrow.setPen(pen)
        graphics.arrow.setBrush(QBrush(graphics.base_color))
        graphics.arrow.setOpacity(opacity)
        if graphics.label is not None:
            graphics.label.setOpacity(1.0 if highlighted else (0.28 if dimmed else 1.0))

    def _draw_connection(self, spec: ConnectionSpec) -> None:
        if spec.source not in self._items or spec.target not in self._items:
            return
        a = self._items[spec.source].sceneBoundingRect()
        b = self._items[spec.target].sceneBoundingRect()
        if spec.feedback:
            # Feedback sources are normally drawn below the controlled path.
            # Route the return in the free gap immediately above the sensor,
            # then rise vertically at the destination.  This keeps long PFC
            # feedback lines out of the parameter/sensor blocks and avoids the
            # old "big loop below everything" crossings.
            feedback_index = sum(1 for g in self._connection_graphics if g.spec.feedback)
            if a.top() > b.bottom() + 20.0:
                p1 = QPointF(a.center().x(), a.top())
                p4 = QPointF(b.center().x(), b.bottom())
                proposed = a.top() - 18.0 - 14.0 * feedback_index
                bus_y = max(b.bottom() + 8.0, proposed)
                if bus_y < a.top() - 2.0:
                    points = [
                        p1,
                        QPointF(p1.x(), bus_y),
                        QPointF(p4.x(), bus_y),
                        p4,
                    ]
                else:
                    midy = (a.top() + b.bottom()) / 2.0
                    points = [p1, QPointF(p1.x(), midy), QPointF(p4.x(), midy), p4]
            else:
                # Fallback for a feedback source that is not below its target.
                p1 = QPointF(a.center().x(), a.bottom())
                p4 = QPointF(b.center().x(), b.bottom())
                y = max(a.bottom(), b.bottom()) + 46.0 + 28.0 * feedback_index
                points = [p1, QPointF(p1.x(), y), QPointF(p4.x(), y), p4]
        else:
            # Choose a clear orthogonal path.  Same-row blocks connect left/right;
            # feedback/detail rows use vertical ports and a short dog-leg.
            same_row = abs(a.center().y() - b.center().y()) < min(a.height(), b.height()) * 0.55
            if b.center().x() >= a.center().x() and same_row:
                p1 = QPointF(a.right(), a.center().y())
                p4 = QPointF(b.left(), b.center().y())
            elif b.center().x() < a.center().x() and same_row:
                p1 = QPointF(a.left(), a.center().y())
                p4 = QPointF(b.right(), b.center().y())
            elif b.center().y() >= a.center().y():
                p1 = QPointF(a.center().x(), a.bottom())
                p4 = QPointF(b.center().x(), b.top())
            else:
                p1 = QPointF(a.center().x(), a.top())
                p4 = QPointF(b.center().x(), b.bottom())
            midx = (p1.x() + p4.x()) / 2.0
            midy = (p1.y() + p4.y()) / 2.0
            if abs(p1.y() - p4.y()) < 1.0:
                points = [p1, p4]
            elif abs(p1.x() - p4.x()) < 1.0:
                points = [p1, p4]
            elif abs(p1.x() - p4.x()) >= abs(p1.y() - p4.y()):
                points = [p1, QPointF(midx, p1.y()), QPointF(midx, p4.y()), p4]
            else:
                points = [p1, QPointF(p1.x(), midy), QPointF(p4.x(), midy), p4]

        base_color = QColor("#0e9384" if spec.feedback else "#175cd3")
        pen = QPen(base_color, 1.8)
        pen.setCosmetic(True)
        line_items = []
        for p, q in zip(points[:-1], points[1:]):
            line_items.append(self.scene.addLine(p.x(), p.y(), q.x(), q.y(), pen))

        # Arrow head at target.
        p_from, p_to = points[-2], points[-1]
        dx, dy = p_to.x() - p_from.x(), p_to.y() - p_from.y()
        norm = max((dx * dx + dy * dy) ** 0.5, 1e-9)
        ux, uy = dx / norm, dy / norm
        px, py = -uy, ux
        size = 10.0
        tip = p_to
        base = QPointF(tip.x() - ux * size, tip.y() - uy * size)
        poly = QPolygonF([
            tip,
            QPointF(base.x() + px * size * 0.45, base.y() + py * size * 0.45),
            QPointF(base.x() - px * size * 0.45, base.y() - py * size * 0.45),
        ])
        arrow = self.scene.addPolygon(poly, pen, QBrush(base_color))
        arrow.setZValue(2)
        label_item: QGraphicsSimpleTextItem | None = None
        if spec.label:
            label_item = QGraphicsSimpleTextItem(spec.label)
            font = QFont()
            font.setPointSize(9)
            label_item.setFont(font)
            label_item.setBrush(QBrush(QColor("#344054")))
            mid = points[len(points) // 2]
            label_item.setPos(mid.x() + 5.0, mid.y() - 21.0)
            self.scene.addItem(label_item)

        self._connection_graphics.append(_ConnectionGraphics(
            spec=spec,
            lines=line_items,
            arrow=arrow,
            label=label_item,
            base_color=base_color,
        ))


__all__ = ["BlockSpec", "ConnectionSpec", "ControlBlockDiagram"]
