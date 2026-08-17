"""GitHub Release 更新检查:后台联网对比最新版本,支持手动/自动检查。

仅使用标准库 (urllib) 完成网络请求,不引入额外依赖。
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass

from PySide6.QtCore import QObject, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QLabel, QMessageBox, QToolButton

from llc_design.gui import theme

#: 与 pyproject.toml / GitHub tag 保持一致的当前版本号。
APP_VERSION = "7.4.0"

GITHUB_REPO = "yangshuai2022-star/llc_design_tool_v1"
RELEASES_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
RELEASES_PAGE_URL = f"https://github.com/{GITHUB_REPO}/releases/latest"
NETWORK_TIMEOUT_S = 10.0

_VERSION_RE = re.compile(r"^[vV]?(\d+)\.(\d+)\.(\d+)")


def parse_version(tag: str) -> tuple[int, int, int] | None:
    """解析 'v7.3.0' 形式的 tag;无法解析时返回 None。"""
    match = _VERSION_RE.match(tag.strip())
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


@dataclass
class ReleaseInfo:
    """GitHub 最新 Release 的摘要信息。"""

    tag_name: str
    name: str
    body: str
    html_url: str
    published_at: str
    is_newer: bool


def fetch_latest_release(timeout: float = NETWORK_TIMEOUT_S) -> ReleaseInfo:
    """请求 GitHub API 获取最新 Release,失败时抛出异常。"""
    request = urllib.request.Request(
        RELEASES_API_URL,
        headers={
            "User-Agent": "PowerDesignToolkit",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))

    tag_name = str(data.get("tag_name", ""))
    latest = parse_version(tag_name)
    current = parse_version(APP_VERSION)
    if latest is None:
        is_newer = bool(tag_name)
    elif current is None:
        is_newer = True
    else:
        is_newer = latest > current

    return ReleaseInfo(
        tag_name=tag_name,
        name=str(data.get("name") or tag_name),
        body=str(data.get("body") or ""),
        html_url=str(data.get("html_url") or RELEASES_PAGE_URL),
        published_at=str(data.get("published_at") or ""),
        is_newer=is_newer,
    )


class UpdateCheckThread(QThread):
    """在后台线程执行网络检查,结果通过信号回传 GUI 线程。"""

    finished_sig = Signal(object)
    error_sig = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

    def run(self) -> None:
        try:
            info = fetch_latest_release()
        except Exception as exc:  # noqa: BLE001 - 网络错误统一上报
            self.error_sig.emit(str(exc))
            return
        self.finished_sig.emit(info)


def check_for_updates(parent: QObject, notify_up_to_date: bool) -> UpdateCheckThread:
    """启动一次异步更新检查。

    ``notify_up_to_date=True`` 时手动检查(总是弹窗反馈);
    ``False`` 时自动检查(仅在新版本时弹窗,失败静默)。
    """
    holder = getattr(parent, "update_check_threads", None)
    if holder is None:
        holder = []
        parent.update_check_threads = holder  # type: ignore[attr-defined]

    thread = UpdateCheckThread(parent)
    thread.finished_sig.connect(
        lambda info: _on_check_result(parent, info, notify_up_to_date))
    thread.error_sig.connect(
        lambda message: _on_check_error(parent, message, notify_up_to_date))
    thread.finished.connect(thread.deleteLater)
    holder.append(thread)
    thread.start()
    return thread


def add_toolbar_right_side(toolbar, main_window) -> None:
    """在工具栏最右侧追加联系方式与"检查更新"按钮。

    ``toolbar`` 必须已添加过可伸缩 spacer,此函数只在 spacer 之后追加内容。
    """
    muted = theme.active_theme().text_muted
    email_label = QLabel(
        f'<a href="mailto:maileyang@qq.com" style="color:{muted};'
        f'text-decoration:none;font-size:12px;">maileyang@qq.com</a>'
    )
    email_label.setToolTip("联系邮箱(点击发送邮件)")
    email_label.setOpenExternalLinks(True)
    toolbar.addWidget(email_label)

    wechat_label = QLabel("微信: maileyang")
    wechat_label.setToolTip("微信号: maileyang")
    wechat_label.setStyleSheet(
        f"color:{muted};font-size:12px;padding:0 6px;background:transparent;")
    toolbar.addWidget(wechat_label)

    update_button = QToolButton()
    update_button.setText("检查更新")
    update_button.setToolTip(f"检查 GitHub 是否有新版本(当前 {APP_VERSION})")
    update_button.clicked.connect(
        lambda: check_for_updates(main_window, notify_up_to_date=True))
    toolbar.addWidget(update_button)


def _on_check_result(parent: QObject, info: ReleaseInfo, notify_up_to_date: bool) -> None:
    if info.is_newer:
        body_preview = (info.body.strip() or "(该版本未附带发布说明)")
        if len(body_preview) > 1200:
            body_preview = body_preview[:1200] + "…"
        box = QMessageBox(parent)
        box.setWindowTitle("发现新版本")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText(
            f"<h3>发现新版本 {info.tag_name}</h3>"
            f"<p>当前版本 {APP_VERSION} · 最新版本 {info.tag_name}<br>"
            f"发布时间: {info.published_at}</p>"
        )
        box.setInformativeText(body_preview)
        open_button = box.addButton("打开下载页", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("知道了", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is open_button:
            QDesktopServices.openUrl(QUrl(info.html_url))
        return

    if notify_up_to_date:
        QMessageBox.information(
            parent, "检查更新", f"当前已是最新版本 {APP_VERSION}。")


def _on_check_error(parent: QObject, message: str, notify_up_to_date: bool) -> None:
    if notify_up_to_date:
        QMessageBox.warning(
            parent, "检查更新失败",
            "无法连接到 GitHub,请检查网络后重试。\n"
            f"<small>{message}</small>",
        )
        return
    status_bar = getattr(parent, "statusBar", None)
    if callable(status_bar):
        status_bar().showMessage("检查更新失败(网络不可用)", 5000)


__all__ = [
    "APP_VERSION",
    "ReleaseInfo",
    "UpdateCheckThread",
    "add_toolbar_right_side",
    "check_for_updates",
    "fetch_latest_release",
    "parse_version",
]
