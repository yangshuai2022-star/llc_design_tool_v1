from html.parser import HTMLParser
from pathlib import Path
import unittest


INDEX_HTML = Path(__file__).parents[1] / "static" / "index.html"


class _ShellParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


def _elements():
    parser = _ShellParser()
    parser.feed(INDEX_HTML.read_text(encoding="utf-8"))
    return parser.elements


class FrontendAccessibilityTests(unittest.TestCase):
    def test_keyboard_users_can_skip_navigation_to_main_content(self):
        elements = _elements()
        self.assertTrue(
            any(
                tag == "a"
                and "skip-link" in attrs.get("class", "").split()
                and attrs.get("href") == "#mainContent"
                for tag, attrs in elements
            )
        )
        self.assertTrue(
            any(
                tag == "main"
                and attrs.get("id") == "mainContent"
                and attrs.get("tabindex") == "-1"
                for tag, attrs in elements
            )
        )

    def test_server_connection_feedback_is_announced(self):
        elements = _elements()
        self.assertTrue(
            any(
                tag == "div"
                and attrs.get("class") == "server-pill"
                and attrs.get("role") == "status"
                and attrs.get("aria-live") == "polite"
                for tag, attrs in elements
            )
        )

    def test_overflowing_tables_are_keyboard_scroll_regions(self):
        wrappers = [
            attrs
            for tag, attrs in _elements()
            if tag == "div" and "table-wrap" in attrs.get("class", "").split()
        ]
        self.assertGreater(len(wrappers), 0)
        for attrs in wrappers:
            self.assertEqual(attrs.get("tabindex"), "0")
            self.assertEqual(attrs.get("role"), "region")
            self.assertTrue(attrs.get("aria-label"))
