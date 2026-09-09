"""Regression tests for the non-intrusive GUI update checker."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject

from llc_design.gui import updater


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("v9.1.0", "9.1.0", False),
        ("v9.1.0", "9.1.1", True),
        ("v9.1.0-rc.1", "v9.1.0", True),
        ("v9.1.0-rc.10", "v9.1.0-rc.2", False),
    ],
)
def test_semantic_version_comparison(left: str, right: str, expected: bool) -> None:
    assert (updater.parse_version(left) < updater.parse_version(right)) is expected


@pytest.mark.parametrize("tag", ["v9.1", "release-9.1.0", "v09.1.0", "v9.1.0-01"])
def test_parse_version_rejects_non_semver_tags(tag: str) -> None:
    assert updater.parse_version(tag) is None


def test_release_equal_to_current_is_not_an_update() -> None:
    payload = {"tag_name": "v9.1.0", "html_url": "https://example.test/release"}
    with patch.object(updater.urllib.request, "urlopen", return_value=_Response(payload)):
        assert updater.fetch_latest_release().is_newer is False


@pytest.mark.parametrize("tag", ["v9.0.9", "unparseable-release"])
def test_older_or_unparseable_release_is_not_an_update(tag: str) -> None:
    with patch.object(
        updater.urllib.request,
        "urlopen",
        return_value=_Response({"tag_name": tag}),
    ):
        assert updater.fetch_latest_release().is_newer is False


def test_automatic_check_runs_once_per_session() -> None:
    updater._auto_check_started = False
    parent = QObject()
    with patch.object(updater, "UpdateCheckThread") as thread_cls:
        thread = thread_cls.return_value
        thread.finished_sig.connect = lambda *_args: None
        thread.error_sig.connect = lambda *_args: None
        thread.finished.connect = lambda *_args: None
        assert updater.check_for_updates(parent, notify_up_to_date=False) is thread
        assert updater.check_for_updates(parent, notify_up_to_date=False) is None
        assert thread_cls.call_count == 1


def test_automatic_network_failure_is_silent() -> None:
    with patch.object(updater.QMessageBox, "warning") as warning:
        updater._on_check_error(QObject(), "offline", notify_up_to_date=False)
    warning.assert_not_called()


def test_dismissed_release_does_not_prompt_again_automatically() -> None:
    info = updater.ReleaseInfo("v9.2.0", "V9.2", "", "", "", True)
    with (
        patch.object(updater, "_dismissed_release", return_value="v9.2.0"),
        patch.object(updater.QMessageBox, "exec") as show,
    ):
        updater._on_check_result(QObject(), info, notify_up_to_date=False)

    show.assert_not_called()
