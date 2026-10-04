"""Playwright backend restricted to the bundled loopback benchmark."""

import time
from urllib.parse import urlparse


_ALLOWED_HOSTS = {"localhost", "127.0.0.1", "::1"}


def assert_loopback_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in _ALLOWED_HOSTS:
        raise RuntimeError(
            "local benchmark browser backend only permits localhost/loopback URLs"
        )


class LocalOnlyPlaywrightBackend:
    """Dispatch real Chromium mouse events only on a loopback benchmark page."""

    def __init__(self, page) -> None:
        self.page = page
        self.position = (0, 0)

    def _guard(self) -> None:
        assert_loopback_url(self.page.url)

    @staticmethod
    def _sleep(duration_ms: int) -> None:
        if duration_ms > 0:
            time.sleep(duration_ms / 1000.0)

    def move_to(self, x: int, y: int, duration_ms: int = 0) -> None:
        self._guard()
        self.page.mouse.move(float(x), float(y), steps=1)
        self.position = (int(x), int(y))
        self._sleep(duration_ms)

    def press(self, x: int, y: int) -> None:
        self._guard()
        if self.position != (int(x), int(y)):
            self.page.mouse.move(float(x), float(y), steps=1)
            self.position = (int(x), int(y))
        self.page.mouse.down(button="left")

    def release(self, x: int, y: int) -> None:
        self._guard()
        if self.position != (int(x), int(y)):
            self.page.mouse.move(float(x), float(y), steps=1)
            self.position = (int(x), int(y))
        self.page.mouse.up(button="left")

    def click(self, x: int, y: int) -> None:
        self._guard()
        self.page.mouse.move(float(x), float(y), steps=1)
        self.page.mouse.down(button="left")
        time.sleep(0.04)
        self.page.mouse.up(button="left")
        self.position = (int(x), int(y))
