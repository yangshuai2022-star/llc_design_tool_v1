"""Compact, responsive analog measurement-chain schematic for control pages."""
from __future__ import annotations

from PySide6.QtCore import Qt, QRectF, QPointF, QSize
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPolygonF
from PySide6.QtWidgets import QWidget, QSizePolicy


class AnalogSenseSchematic(QWidget):
    """Draw a five-stage sensing chain without requiring a wide panel.

    Wide layouts use a single horizontal row.  Narrow parameter inspectors use
    a 3+2 wrapped flow so the schematic remains legible instead of being clipped.
    """

    def __init__(self, title: str = "Analog sensing chain", parent=None):
        super().__init__(parent)
        self.title = title
        self.source_label = "Measured quantity"
        self.front_label = "Divider / Sensor"
        self.amp_label = "OpAmp"
        self.rc_label = "RADC / CADC"
        self.adc_label = "ADC + Digital"
        # The old minimum height (118 px) was smaller than the wrapped 3+2
        # drawing's actual 150+ px content.  QFormLayout therefore legally
        # compressed the widget and clipped the lower row on both LLC and PFC
        # side inspectors.  Advertise height-for-width so narrow inspectors get
        # the full wrapped schematic while wide panes remain compact.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumHeight(162)
        self.setMaximumHeight(190)


    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        # Wide single-row chain needs only ~105 px.  Narrow 3+2 chain draws the
        # second row at y=105 with 42 px boxes, plus margins/title.
        return 112 if width >= 650 else 162

    def sizeHint(self) -> QSize:
        return QSize(360, self.heightForWidth(360))

    def minimumSizeHint(self) -> QSize:
        return QSize(260, 162)

    def set_labels(self, *, title=None, source=None, front=None, amp=None, rc=None, adc=None):
        if title is not None:
            self.title = title
        if source is not None:
            self.source_label = source
        if front is not None:
            self.front_label = front
        if amp is not None:
            self.amp_label = amp
        if rc is not None:
            self.rc_label = rc
        if adc is not None:
            self.adc_label = adc
        self.update()

    @staticmethod
    def _arrow(painter: QPainter, p1: QPointF, p2: QPointF) -> None:
        pen = QPen(QColor("#667085"), 1.35)
        painter.setPen(pen)
        painter.drawLine(p1, p2)
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        norm = max((dx * dx + dy * dy) ** 0.5, 1e-9)
        ux, uy = dx / norm, dy / norm
        px, py = -uy, ux
        size = 6.0
        base = QPointF(p2.x() - ux * size, p2.y() - uy * size)
        poly = QPolygonF([
            p2,
            QPointF(base.x() + px * size * 0.45, base.y() + py * size * 0.45),
            QPointF(base.x() - px * size * 0.45, base.y() - py * size * 0.45),
        ])
        painter.setBrush(QBrush(QColor("#667085")))
        painter.drawPolygon(poly)

    def _draw_box(self, painter: QPainter, rect: QRectF, label: str, highlight: bool = False) -> None:
        painter.setPen(QPen(QColor("#5b6b82"), 1.25))
        painter.setBrush(QBrush(QColor("#eef4ff" if highlight else "#f8fafc")))
        painter.drawRoundedRect(rect, 7, 7)
        painter.setPen(QPen(QColor("#344054"), 1.0))
        painter.drawText(
            rect.adjusted(5, 3, -5, -3),
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
            label,
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QPen(QColor("#344054"), 1.0))
        painter.drawText(
            QRectF(8, 3, self.width() - 16, 20),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self.title,
        )

        labels = [
            self.source_label,
            self.front_label,
            self.amp_label,
            self.rc_label,
            self.adc_label,
        ]
        width = float(self.width())

        if width >= 650.0:
            left, right, y, gap = 10.0, width - 10.0, 40.0, 10.0
            box_w = (right - left - gap * 4.0) / 5.0
            box_h = 48.0
            rects = [QRectF(left + i * (box_w + gap), y, box_w, box_h) for i in range(5)]
            for i, (rect, label) in enumerate(zip(rects, labels)):
                self._draw_box(painter, rect, label, highlight=i in (1, 3))
                if i < 4:
                    self._arrow(
                        painter,
                        QPointF(rect.right(), rect.center().y()),
                        QPointF(rects[i + 1].left() - 3.0, rect.center().y()),
                    )
        else:
            # Responsive 3+2 layout for 300–450 px inspectors.
            left, right, gap = 10.0, width - 10.0, 8.0
            top_y, bottom_y = 35.0, 105.0
            box_h = 42.0
            top_w = (right - left - 2.0 * gap) / 3.0
            bottom_w = (right - left - gap) / 2.0
            rects = [
                QRectF(left, top_y, top_w, box_h),
                QRectF(left + top_w + gap, top_y, top_w, box_h),
                QRectF(left + 2.0 * (top_w + gap), top_y, top_w, box_h),
                QRectF(left, bottom_y, bottom_w, box_h),
                QRectF(left + bottom_w + gap, bottom_y, bottom_w, box_h),
            ]
            for i, (rect, label) in enumerate(zip(rects, labels)):
                self._draw_box(painter, rect, label, highlight=i in (1, 3))
            self._arrow(painter, QPointF(rects[0].right(), rects[0].center().y()), QPointF(rects[1].left() - 3, rects[1].center().y()))
            self._arrow(painter, QPointF(rects[1].right(), rects[1].center().y()), QPointF(rects[2].left() - 3, rects[2].center().y()))
            # Fold the flow to the next row, then continue left-to-right.
            self._arrow(painter, QPointF(rects[2].center().x(), rects[2].bottom()), QPointF(rects[3].center().x(), rects[3].top() - 3))
            self._arrow(painter, QPointF(rects[3].right(), rects[3].center().y()), QPointF(rects[4].left() - 3, rects[4].center().y()))

        painter.end()


__all__ = ["AnalogSenseSchematic"]
