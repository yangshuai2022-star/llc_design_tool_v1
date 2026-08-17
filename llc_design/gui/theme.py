"""Theme support for the Power Design Toolkit GUI.

Detects the Windows 11 light/dark appearance and exposes a single token set
used by the main-window stylesheets, the launcher dialog and every inline
label style. Painted diagram widgets (control block diagram, sense schematic,
matplotlib figures) intentionally keep a light canvas in both themes so that
schematics and Bode plots stay readable on a dark surrounding.

Usage::

    from llc_design.gui import theme
    t = theme.apply_app_theme(app)          # call once at startup
    t = theme.active_theme()                 # anywhere afterwards
    label.setStyleSheet(f"color:{t.text_muted};")
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QStyleFactory


@dataclass(frozen=True)
class ThemeTokens:
    window_bg: str
    surface: str
    surface_alt: str
    tab_bg: str
    border: str
    border_card: str
    border_input: str
    text: str
    text_strong: str
    text_muted: str
    accent: str
    accent_border: str
    hover: str
    pressed: str
    focus_border: str
    checked_bg: str
    splitter: str
    statusbar: str
    card_bg: str
    card_bg_alt: str


LIGHT = ThemeTokens(
    window_bg="#f4f6f8",
    surface="#ffffff",
    surface_alt="#f8fafc",
    tab_bg="#f2f4f7",
    border="#d9dee7",
    border_card="#d0d5dd",
    border_input="#b9c2cf",
    text="#344054",
    text_strong="#101828",
    text_muted="#667085",
    accent="#175cd3",
    accent_border="#84adff",
    hover="#eef4ff",
    pressed="#dbe8ff",
    focus_border="#528bff",
    checked_bg="#eaf2ff",
    splitter="#eef1f5",
    statusbar="#ffffff",
    card_bg="#f8fafc",
    card_bg_alt="#fbfcfe",
)

DARK = ThemeTokens(
    window_bg="#1b1f24",
    surface="#232830",
    surface_alt="#2a3038",
    tab_bg="#2a3038",
    border="#353c47",
    border_card="#3a4150",
    border_input="#454d5b",
    text="#c3cbd6",
    text_strong="#e8edf4",
    text_muted="#8b95a7",
    accent="#6ea8ff",
    accent_border="#4f8ff7",
    hover="#2c3644",
    pressed="#34404f",
    focus_border="#528bff",
    checked_bg="#2f3b4d",
    splitter="#2a3038",
    statusbar="#232830",
    card_bg="#2a3038",
    card_bg_alt="#262c34",
)


def _windows_apps_use_light_theme() -> bool | None:
    """Read HKCU\\...\\Personalize\\AppsUseLightTheme. Returns None if unknown."""
    if sys.platform != "win32":
        return None
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return int(value) == 1
    except Exception:
        return None


def _palette_is_dark(app: QApplication) -> bool:
    pal = app.palette()
    w = pal.color(QPalette.ColorRole.Window)
    luminance = (0.2126 * w.red() + 0.7152 * w.green() + 0.0722 * w.blue()) / 255.0
    return luminance < 0.5


def detect_dark(app: QApplication) -> bool:
    """True when the OS application theme is dark."""
    light = _windows_apps_use_light_theme()
    if light is not None:
        return not light
    return _palette_is_dark(app)


def active_theme() -> ThemeTokens:
    """Return the theme chosen at startup (re-detects if run before apply_app_theme)."""
    app = QApplication.instance()
    if app is not None:
        stored = getattr(app, "_power_design_theme", None)
        if isinstance(stored, ThemeTokens):
            return stored
    app = app or QApplication(sys.argv)
    return apply_app_theme(app)


def apply_app_theme(app: QApplication) -> ThemeTokens:
    """Force Fusion style + a palette matching the OS light/dark appearance."""
    app.setStyle(QStyleFactory.create("Fusion"))
    t = DARK if detect_dark(app) else LIGHT
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(t.window_bg))
    pal.setColor(QPalette.ColorRole.Base, QColor(t.surface))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(t.window_bg))
    pal.setColor(QPalette.ColorRole.Text, QColor(t.text_strong))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(t.text))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(t.text_strong))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(t.text_strong))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(t.text_muted))
    pal.setColor(QPalette.ColorRole.Button, QColor(t.surface))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(t.accent))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    disabled = QColor(t.text_muted)
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        pal.setColor(QPalette.ColorGroup.Disabled, role, disabled)
    app.setPalette(pal)
    app._power_design_theme = t  # type: ignore[attr-defined]
    return t


def workspace_stylesheet(t: ThemeTokens) -> str:
    """Shared stylesheet for the LLC and PFC main windows."""
    return f"""
    QMainWindow {{ background: {t.window_bg}; }}
    QToolBar {{ background: {t.surface}; border-bottom: 1px solid {t.border}; spacing: 4px; padding: 3px 6px; }}
    QToolButton {{ padding: 4px 10px; border-radius: 5px; }}
    QToolButton:hover {{ background: {t.hover}; }}
    QDockWidget {{ font-weight: 600; color: {t.text}; }}
    QDockWidget::title {{ background: {t.surface_alt}; border-bottom: 1px solid {t.border}; padding: 7px 10px; }}
    QTabWidget::pane {{ border: 1px solid {t.border}; background: {t.surface}; top: -1px; }}
    QTabBar::tab {{ background: {t.tab_bg}; border: 1px solid {t.border}; padding: 7px 14px; margin-right: 2px; min-height: 20px; }}
    QTabBar::tab:selected {{ background: {t.surface}; color: {t.accent}; border-bottom-color: {t.surface}; font-weight: 600; }}
    QGroupBox {{ background: {t.surface}; border: 1px solid {t.border}; border-radius: 7px; margin-top: 12px; padding-top: 8px; font-weight: 600; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {t.text}; }}
    QDoubleSpinBox, QSpinBox, QComboBox {{ min-height: 25px; padding: 1px 4px; background: {t.surface}; border: 1px solid {t.border_input}; border-radius: 4px; }}
    QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{ border-color: {t.focus_border}; }}
    QPushButton {{ min-height: 29px; padding: 3px 10px; border: 1px solid {t.border_input}; border-radius: 5px; background: {t.surface}; }}
    QPushButton:hover {{ background: {t.hover}; border-color: {t.accent_border}; }}
    QPushButton:pressed {{ background: {t.pressed}; }}
    QPushButton:checked {{ background: {t.checked_bg}; border-color: {t.accent_border}; color: {t.accent}; }}
    QCheckBox {{ spacing: 6px; color: {t.text}; }}
    QPlainTextEdit {{ background: {t.surface}; border: 1px solid {t.border}; }}
    QScrollArea {{ border: none; background: transparent; }}
    QSplitter::handle {{ background: {t.splitter}; }}
    QStatusBar {{ background: {t.statusbar}; border-top: 1px solid {t.border}; }}
    """


def launcher_stylesheet(t: ThemeTokens) -> str:
    """Stylesheet for the workspace selection dialog and its choice buttons."""
    return f"""
    QDialog {{ background: {t.window_bg}; }}
    QLabel {{ color: {t.text_strong}; }}
    QPushButton {{
        font-size: 16px; font-weight: 600; text-align: center;
        padding: 24px; border: 2px solid {t.border_input}; border-radius: 10px;
        background: {t.surface_alt}; color: {t.text_strong};
    }}
    QPushButton:hover {{ background: {t.hover}; border-color: {t.accent}; }}
    QPushButton:pressed {{ background: {t.pressed}; }}
    """


__all__ = [
    "ThemeTokens",
    "LIGHT",
    "DARK",
    "detect_dark",
    "active_theme",
    "apply_app_theme",
    "workspace_stylesheet",
    "launcher_stylesheet",
]