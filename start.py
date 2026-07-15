#!/usr/bin/env python3
"""
start.py — One-click launcher for SignalLab.

Detects the platform, creates the virtual environment, installs dependencies,
walks the user through API-key setup, then launches the dashboard in a browser.

Normally you don't run this directly — double-click one of:
    Windows        start.bat
    macOS          start.command
    Linux          start.sh

But it works fine on its own too:
    python start.py                # dashboard (default)
    python start.py --api          # FastAPI backend instead
    python start.py --configure    # re-open the API-key window, then start
    python start.py --console      # ask for the key in the terminal, no pop-up
    python start.py --reinstall    # rebuild the virtual environment
    python start.py --setup-only   # prepare everything, don't launch
    python start.py --serve        # share on your Tailscale tailnet (HTTPS)

Stdlib only — this file must run before anything is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
STAMP = VENV_DIR / ".signallab-install.json"
DASHBOARD = ROOT / "ui" / "dashboard.py"

IS_WINDOWS = os.name == "nt"
IS_MACOS = sys.platform == "darwin"
MIN_PYTHON = (3, 10)
MAX_TESTED_PYTHON = (3, 13)

sys.path.insert(0, str(ROOT))

# ── Output helpers ───────────────────────────────────────────────────────────


def _supports_colour() -> bool:
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return False
    if IS_WINDOWS:
        try:
            import ctypes

            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
            return True
        except Exception:
            return False
    return True


_C = _supports_colour()
BOLD = "\033[1m" if _C else ""
DIM = "\033[2m" if _C else ""
GREEN = "\033[32m" if _C else ""
YELLOW = "\033[33m" if _C else ""
RED = "\033[31m" if _C else ""
CYAN = "\033[36m" if _C else ""
RESET = "\033[0m" if _C else ""


def say(msg: str = "") -> None:
    print(msg, flush=True)


def step(n: int, total: int, msg: str) -> None:
    say(f"\n{CYAN}[{n}/{total}]{RESET} {BOLD}{msg}{RESET}")


def ok(msg: str) -> None:
    say(f"{GREEN}✅ {msg}{RESET}")


def info(msg: str) -> None:
    say(f"{DIM}   {msg}{RESET}")


def warn(msg: str) -> None:
    say(f"{YELLOW}⚠️  {msg}{RESET}")


class StartupError(Exception):
    """A failure we can explain to a non-technical user."""

    def __init__(self, message: str, fix: str = ""):
        super().__init__(message)
        self.fix = fix


def banner() -> None:
    say()
    say(f"{CYAN}╭────────────────────────────────────────────────────────────╮{RESET}")
    say(f"{CYAN}│{RESET}  {BOLD}📡  SignalLab — Equity Signal Intelligence{RESET}                {CYAN}│{RESET}")
    say(f"{CYAN}│{RESET}  {DIM}Starting up. First run takes a few minutes.{RESET}               {CYAN}│{RESET}")
    say(f"{CYAN}╰────────────────────────────────────────────────────────────╯{RESET}")
    info(f"{platform.system()} {platform.release()} · Python {platform.python_version()}")


# ── Step 1: Python check ─────────────────────────────────────────────────────


def check_python() -> None:
    version = sys.version_info[:2]
    if version < MIN_PYTHON:
        raise StartupError(
            f"SignalLab needs Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer — this is "
            f"Python {version[0]}.{version[1]}.",
            fix="Install a current Python from https://www.python.org/downloads/ and start again.\n"
                "   On Windows, tick 'Add Python to PATH' during install.",
        )
    if version > MAX_TESTED_PYTHON:
        warn(
            f"Python {version[0]}.{version[1]} is newer than the tested range "
            f"(≤ {MAX_TESTED_PYTHON[0]}.{MAX_TESTED_PYTHON[1]}). If installs fail, "
            f"try Python {MAX_TESTED_PYTHON[0]}.{MAX_TESTED_PYTHON[1]}."
        )
    ok(f"Python {platform.python_version()} at {sys.executable}")


# ── Step 2: Secrets ──────────────────────────────────────────────────────────


def ensure_secrets(force: bool, prefer_gui: bool = True) -> None:
    try:
        import setup_secrets
    except ImportError as e:
        raise StartupError(
            "Couldn't find setup_secrets.py next to start.py.",
            fix="Re-download the project — a file is missing.",
        ) from e

    if setup_secrets.is_configured() and not force:
        ok("API key found in .env")
        info("To change it: re-run with --configure, or run setup_secrets.py")
        return

    if not force:
        info("No API key configured yet — let's set that up now.")

    if not setup_secrets.configure(force=force, prefer_gui=prefer_gui):
        raise StartupError(
            "Setup was cancelled — no API key was saved.",
            fix="Start SignalLab again and paste a working OpenAI key when the window appears.\n"
                f"   Get one at {setup_secrets.KEY_SIGNUP_URL}",
        )


# ── Step 3: Virtual environment ──────────────────────────────────────────────


def venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def ensure_venv(reinstall: bool) -> Path:
    py = venv_python()

    if reinstall and VENV_DIR.exists():
        import shutil

        info("Removing the existing virtual environment…")
        shutil.rmtree(VENV_DIR, ignore_errors=True)

    if py.exists():
        ok(f"Virtual environment ready ({VENV_DIR.name})")
        return py

    info("Creating an isolated virtual environment (.venv) — about 10 seconds…")
    result = subprocess.run(
        [sys.executable, "-m", "venv", str(VENV_DIR)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not py.exists():
        detail = (result.stderr or result.stdout or "").strip()
        fix = "Install the venv module for your Python."
        if "ensurepip" in detail or "python3-venv" in detail:
            fix = (
                "Your Python is missing the venv module. On Debian/Ubuntu run:\n"
                f"   sudo apt install python{sys.version_info.major}.{sys.version_info.minor}-venv"
            )
        raise StartupError(f"Couldn't create the virtual environment.\n{DIM}{detail}{RESET}", fix=fix)

    ok("Virtual environment created")
    return py


# ── Step 4: Dependencies ─────────────────────────────────────────────────────


def _requirements_fingerprint() -> str:
    return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()[:16]


def _already_installed() -> bool:
    if not STAMP.exists():
        return False
    try:
        stamp = json.loads(STAMP.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        stamp.get("requirements") == _requirements_fingerprint()
        and stamp.get("python") == platform.python_version()
    )


def install_deps(py: Path, force: bool) -> None:
    if not REQUIREMENTS.exists():
        raise StartupError("requirements.txt is missing.", fix="Re-download the project.")

    if _already_installed() and not force:
        ok("Dependencies already installed and up to date")
        return

    info("Installing dependencies. First run downloads ~500MB and takes 2–5 minutes.")
    info("This only happens once — later starts are instant.")
    say()

    subprocess.run(
        [str(py), "-m", "pip", "install", "--upgrade", "pip", "--quiet", "--disable-pip-version-check"],
        check=False,
    )

    result = subprocess.run(
        [str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS), "--disable-pip-version-check"],
    )
    if result.returncode != 0:
        raise StartupError(
            "Dependency installation failed (see the pip output above).",
            fix="Common causes: no internet connection, a corporate proxy/firewall, or a "
                "Python version outside 3.10–3.13.\n"
                "   To start clean, delete the .venv folder and run this again.",
        )

    STAMP.write_text(
        json.dumps({"requirements": _requirements_fingerprint(), "python": platform.python_version()}),
        encoding="utf-8",
    )
    say()
    ok("Dependencies installed")


# ── Step 5: Launch ───────────────────────────────────────────────────────────


def _tailscale_bin() -> str | None:
    """
    Find the tailscale CLI. On macOS the Mac App Store / Standalone builds do
    not put it on PATH — the binary lives inside the app bundle.
    """
    import shutil

    found = shutil.which("tailscale")
    if found:
        return found
    for candidate in (
        "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
        "/usr/local/bin/tailscale",
        "/opt/homebrew/bin/tailscale",
    ):
        if Path(candidate).exists():
            return candidate
    return None


def _tailnet_url(ts: str) -> str | None:
    """Resolve this machine's MagicDNS name, e.g. rashmis-macbook-air.tailnet.ts.net"""
    try:
        out = subprocess.run([ts, "status", "--json"], capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return None
        name = json.loads(out.stdout).get("Self", {}).get("DNSName", "").rstrip(".")
        return f"https://{name}" if name else None
    except Exception:
        return None


def enable_tailscale_serve(port: int = 8501) -> str | None:
    """
    Publish the dashboard to the tailnet over HTTPS. Returns the URL, or None.

    Serve proxies from the tailnet to 127.0.0.1, so the dashboard stays bound to
    localhost and is unreachable from the LAN — only tailnet members get in, and
    Tailscale terminates TLS with a real certificate.
    """
    ts = _tailscale_bin()
    if not ts:
        warn("Tailscale not found — starting on localhost only.")
        info("Install it from https://tailscale.com/download, then run with --serve again.")
        return None

    state = subprocess.run([ts, "status"], capture_output=True, text=True)
    if state.returncode != 0:
        warn("Tailscale is installed but not logged in / running.")
        info(f"Run: {ts} up")
        return None

    result = subprocess.run([ts, "serve", "--bg", str(port)], capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        warn(f"Couldn't enable Tailscale Serve: {detail[:200]}")
        info("If it asks you to enable HTTPS for the tailnet, follow the link it printed, then retry.")
        return None

    url = _tailnet_url(ts)
    ok("Tailscale Serve enabled — the dashboard is on your tailnet")
    info("Serve persists across reboots. Turn it off with: "
         f"{ts} serve --bg {port} off")
    return url


def _silence_streamlit_prompts() -> None:
    """Skip Streamlit's first-run email prompt, which blocks an unattended start."""
    creds = Path.home() / ".streamlit" / "credentials.toml"
    if creds.exists():
        return
    try:
        creds.parent.mkdir(parents=True, exist_ok=True)
        creds.write_text('[general]\nemail = ""\n', encoding="utf-8")
    except Exception:
        pass


def launch_dashboard(py: Path, serve: bool = False, headless: bool = False) -> int:
    if not DASHBOARD.exists():
        raise StartupError(f"Dashboard not found at {DASHBOARD}.", fix="Re-download the project.")

    _silence_streamlit_prompts()
    env = dict(os.environ, STREAMLIT_BROWSER_GATHER_USAGE_STATS="false", PYTHONPATH=str(ROOT))

    tailnet_url = enable_tailscale_serve(8501) if serve else None

    say()
    say(f"{GREEN}{'═' * 62}{RESET}")
    say(f"{BOLD}  🚀  SignalLab is starting…{RESET}")
    if tailnet_url:
        say(f"{BOLD}  On your tailnet:  {tailnet_url}{RESET}")
        say(f"{DIM}  Anyone signed in to your tailnet can open that URL.{RESET}")
    say(f"{DIM}  On this Mac:      http://localhost:8501{RESET}")
    say(f"{DIM}  Keep this window open while you use the app.{RESET}")
    say(f"{DIM}  Press Ctrl+C here (or just close this window) to stop.{RESET}")
    say(f"{GREEN}{'═' * 62}{RESET}")
    say()

    cmd = [
        str(py), "-m", "streamlit", "run", str(DASHBOARD),
        "--server.port=8501",
        # Bind to loopback only. Streamlit's default is every interface, which
        # would put the dashboard on whatever wifi this laptop joins. Tailscale
        # Serve proxies in from the tailnet, so localhost is all we need — and
        # it stops anyone on the LAN bypassing Serve to spoof identity headers.
        "--server.address=127.0.0.1",
    ]
    if headless or serve:
        cmd.append("--server.headless=true")   # don't pop a browser on a host machine

    try:
        return subprocess.run(cmd, env=env, cwd=str(ROOT)).returncode
    except KeyboardInterrupt:
        say(f"\n{DIM}SignalLab stopped. Run the start file again any time.{RESET}")
        return 0


def launch_api(py: Path) -> int:
    say()
    say(f"{GREEN}{'═' * 62}{RESET}")
    say(f"{BOLD}  🌐  SignalLab API starting on http://localhost:8000{RESET}")
    say(f"{DIM}  Interactive docs: http://localhost:8000/docs{RESET}")
    say(f"{DIM}  Press Ctrl+C to stop.{RESET}")
    say(f"{GREEN}{'═' * 62}{RESET}")
    say()

    env = dict(os.environ, PYTHONPATH=str(ROOT))
    try:
        return subprocess.run(
            [str(py), "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"],
            env=env,
            cwd=str(ROOT),
        ).returncode
    except KeyboardInterrupt:
        say(f"\n{DIM}API stopped.{RESET}")
        return 0


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Start SignalLab.", add_help=True)
    parser.add_argument("--api", action="store_true", help="Start the FastAPI backend instead of the dashboard.")
    parser.add_argument("--configure", action="store_true", help="Re-open the API-key setup window before starting.")
    parser.add_argument("--console", action="store_true", help="Ask for the API key in this terminal, no pop-up.")
    parser.add_argument("--reinstall", action="store_true", help="Delete .venv and reinstall everything.")
    parser.add_argument("--setup-only", action="store_true", help="Set everything up but don't launch.")
    parser.add_argument("--serve", action="store_true", help="Publish to your Tailscale tailnet over HTTPS.")
    parser.add_argument("--headless", action="store_true", help="Don't open a browser (for host machines).")
    args = parser.parse_args()

    banner()
    total = 4 if args.setup_only else 5

    step(1, total, "Checking Python")
    check_python()

    step(2, total, "Checking your API key")
    ensure_secrets(force=args.configure, prefer_gui=not args.console)

    step(3, total, "Preparing the virtual environment")
    py = ensure_venv(reinstall=args.reinstall)

    step(4, total, "Installing dependencies")
    install_deps(py, force=args.reinstall)

    if args.setup_only:
        say()
        ok("Setup complete. Double-click the start file to launch SignalLab.")
        return 0

    step(5, total, "Launching")
    return launch_api(py) if args.api else launch_dashboard(py, serve=args.serve, headless=args.headless)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except StartupError as e:
        say()
        say(f"{RED}{'═' * 62}{RESET}")
        say(f"{RED}❌ {e}{RESET}")
        if e.fix:
            say(f"\n{YELLOW}How to fix:{RESET}\n   {e.fix}")
        say(f"{RED}{'═' * 62}{RESET}")
        sys.exit(1)
    except KeyboardInterrupt:
        say(f"\n{DIM}Cancelled.{RESET}")
        sys.exit(130)
