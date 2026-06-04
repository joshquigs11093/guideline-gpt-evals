"""Capture dashboard screenshots for the README.

Launches the Streamlit dashboard headless, drives it with Playwright, and writes
one PNG per page to ``docs/images/``. Requires Playwright + a browser:

    uv pip install playwright
    python -m playwright install chromium
    python scripts/capture_screenshots.py

The dashboard reads the committed (synthetic) results, so no API keys are needed.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
IMAGES_DIR = REPO / "docs" / "images"
DASHBOARD = REPO / "src" / "eval_harness" / "ui" / "dashboard.py"

# (page label in the sidebar nav, output filename)
PAGES = [
    ("Overview", "overview.png"),
    ("Experiments", "experiments.png"),
    ("Compare", "compare.png"),
    ("Question Explorer", "question_explorer.png"),
    ("Failure Analysis", "failure_analysis.png"),
    ("Methodology", "methodology.png"),
]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_healthy(port: int, timeout: float = 60.0) -> None:
    url = f"http://localhost:{port}/_stcore/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:  # noqa: BLE001 - server not up yet
            time.sleep(1)
    raise SystemExit("dashboard did not become healthy in time")


def main() -> None:
    from playwright.sync_api import sync_playwright

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(DASHBOARD),
            "--server.headless",
            "true",
            "--server.port",
            str(port),
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=str(REPO),
    )
    try:
        _wait_healthy(port)
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1024})
            for label, filename in PAGES:
                page.goto(f"http://localhost:{port}/", wait_until="networkidle")
                link = page.get_by_role("link", name=label)
                if link.count():
                    link.first.click()
                    page.wait_for_load_state("networkidle")
                page.wait_for_timeout(2500)  # let charts render
                page.screenshot(path=str(IMAGES_DIR / filename), full_page=True)
                print(f"captured {filename}")
            browser.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
