#!/usr/bin/env python3
"""
setup_secrets.py — Secrets configuration for SignalLab.

Writes a `.env` file in the project root containing your API key and model
preferences. This is the only file you ever need to edit by hand, and this
script means you don't even have to do that.

Two front-ends over the same logic:
  • A pop-up window (tkinter) — the default, used when a display is available.
  • A console wizard — automatic fallback for headless machines, SSH sessions,
    or Linux boxes without python3-tk installed.

Usage:
    python setup_secrets.py              # pop-up (falls back to console)
    python setup_secrets.py --console    # force the console wizard
    python setup_secrets.py --check      # exit 0 if configured, 1 if not
    python setup_secrets.py --show       # print current config (key masked)

Stdlib only — runs before any dependencies are installed.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import urllib.error
import urllib.request
from getpass import getpass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"

IS_WINDOWS = os.name == "nt"
IS_MACOS = sys.platform == "darwin"

KEY_SIGNUP_URL = "https://platform.openai.com/api-keys"

# ── Terminal colours (disabled where unsupported) ────────────────────────────

def _supports_colour() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    if os.name == "nt":
        return os.environ.get("WT_SESSION") or os.environ.get("TERM") or _enable_win_ansi()
    return True


def _enable_win_ansi() -> bool:
    try:
        import ctypes

        k = ctypes.windll.kernel32
        k.SetConsoleMode(k.GetStdHandle(-11), 7)
        return True
    except Exception:
        return False


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


def ok(msg: str) -> None:
    say(f"{GREEN}✅ {msg}{RESET}")


def warn(msg: str) -> None:
    say(f"{YELLOW}⚠️  {msg}{RESET}")


def err(msg: str) -> None:
    say(f"{RED}❌ {msg}{RESET}")


def rule(title: str = "") -> None:
    say(f"\n{CYAN}{'─' * 66}{RESET}")
    if title:
        say(f"{BOLD}{title}{RESET}")
        say(f"{CYAN}{'─' * 66}{RESET}")


# ── Field definitions ────────────────────────────────────────────────────────


class Field:
    """One line in the .env file."""

    def __init__(
        self,
        key: str,
        prompt: str,
        default: str = "",
        secret: bool = False,
        required: bool = False,
        choices: list[str] | None = None,
        help_text: str = "",
        comment: str = "",
        validator: Callable[[str], str | None] | None = None,
    ):
        self.key = key
        self.prompt = prompt
        self.default = default
        self.secret = secret
        self.required = required
        self.choices = choices
        self.help_text = help_text
        self.comment = comment
        self.validator = validator


def _validate_openai_key(value: str) -> str | None:
    if not value.startswith("sk-"):
        return "OpenAI keys normally start with 'sk-'. Double-check you pasted the whole key."
    if len(value) < 20:
        return "That looks too short to be a complete key."
    return None


FIELDS: list[Field] = [
    Field(
        key="OPENAI_API_KEY",
        prompt="OpenAI API key",
        secret=True,
        required=True,
        help_text=(
            "Get one at https://platform.openai.com/api-keys\n"
            "   You need a payment method on the OpenAI account. A typical\n"
            "   SignalLab run costs about $0.10–$0.30 with gpt-4o-mini."
        ),
        comment="Used for both reasoning (signal agents) and embeddings.",
        validator=_validate_openai_key,
    ),
    Field(
        key="OPENAI_MODEL",
        prompt="Reasoning model",
        default="gpt-4o-mini",
        choices=["gpt-4o-mini", "gpt-4o"],
        help_text="gpt-4o-mini is cheaper and faster. gpt-4o reasons better but costs ~4x more.",
        comment="gpt-4o-mini = cheap + fast · gpt-4o = higher quality reasoning.",
    ),
    Field(
        key="EMBED_MODEL",
        prompt="Embedding model",
        default="text-embedding-3-small",
        choices=["text-embedding-3-small", "text-embedding-3-large"],
        help_text="Leave the default unless you know you want larger embeddings.",
        comment="Used to index documents into ChromaDB.",
    ),
]

# Advanced knobs — written to .env with defaults, never prompted for.
ADVANCED: list[tuple[str, str, str]] = [
    ("OPENAI_TEMPERATURE", "0.0", "Keep at 0.0 for deterministic, evidence-grounded signals."),
    ("NSE_BSE_RESULT_LAG_DAYS", "90", "Days after quarter-end to keep looking for results/decks/transcripts."),
    ("NSE_BSE_ANNUAL_LAG_DAYS", "200", "Annual reports land much later (around the AGM)."),
    ("NSE_BSE_MAX_DOCS_PER_QUARTER", "15", "Cap on documents ingested per quarter."),
    ("NSE_BSE_MAX_PAGES", "5", "Announcement listing pages to page through per exchange."),
]

ALL_KNOWN_KEYS = {f.key for f in FIELDS} | {k for k, _, _ in ADVANCED}


# ── .env read / write ────────────────────────────────────────────────────────


def read_env(path: Path = ENV_PATH) -> dict[str, str]:
    """Parse an existing .env into a dict. Tolerant of comments and blanks."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        values[key.strip()] = val
    return values


def write_env(values: dict[str, str], path: Path = ENV_PATH) -> None:
    """Render a commented .env file, preserving any unknown keys the user added."""
    lines = [
        "# ─────────────────────────────────────────────────────────────────────",
        "#  SignalLab secrets — generated by `python setup_secrets.py`",
        "#  This file is git-ignored. Never commit it or share it.",
        "#  Re-run the wizard any time to change these values.",
        "# ─────────────────────────────────────────────────────────────────────",
        "",
        "# ── Required ──────────────────────────────────────────────────────────",
    ]
    for field in FIELDS:
        if field.comment:
            lines.append(f"# {field.comment}")
        lines.append(f"{field.key}={values.get(field.key, field.default)}")
        lines.append("")

    lines.append("# ── Advanced (safe defaults — edit only if you know why) ──────────────")
    for key, default, comment in ADVANCED:
        lines.append(f"# {comment}")
        lines.append(f"{key}={values.get(key, default)}")
        lines.append("")

    extras = {k: v for k, v in values.items() if k not in ALL_KNOWN_KEYS}
    if extras:
        lines.append("# ── Your own additions ────────────────────────────────────────────────")
        for key, val in extras.items():
            lines.append(f"{key}={val}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")

    # Owner read/write only, where the OS supports it.
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass


def mask(value: str) -> str:
    if not value:
        return "(not set)"
    if len(value) <= 12:
        return value[:2] + "…" + value[-2:]
    return f"{value[:7]}…{value[-4:]}"


# ── Key verification ─────────────────────────────────────────────────────────


def verify_openai_key(key: str, timeout: int = 15) -> tuple[bool, str]:
    """Call the OpenAI /v1/models endpoint. Returns (verified, message)."""
    req = urllib.request.Request(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            n = len(data.get("data", []))
            return True, f"Key is valid — {n} models available on this account."
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return False, "OpenAI rejected this key (401 Unauthorized). Check for typos or a revoked key."
        if e.code == 429:
            return True, "Key accepted, but the account is rate-limited or out of quota (429)."
        return False, f"OpenAI returned HTTP {e.code}. Key may still be fine."
    except urllib.error.URLError as e:
        return False, f"Couldn't reach OpenAI to check the key ({e.reason}). Skipping verification."
    except Exception as e:  # noqa: BLE001
        return False, f"Verification skipped ({e})."


# ── Prompting ────────────────────────────────────────────────────────────────


def ask(field: Field, current: str) -> str:
    say()
    say(f"{BOLD}{field.prompt}{RESET}")
    if field.help_text:
        say(f"{DIM}   {field.help_text}{RESET}")
    if field.choices:
        say(f"{DIM}   Options: {' · '.join(field.choices)}{RESET}")

    existing = current or field.default
    if field.secret and current:
        hint = f" [keep current: {mask(current)}]"
    elif existing:
        hint = f" [{existing}]"
    else:
        hint = ""

    while True:
        try:
            if field.secret:
                say(f"{DIM}   (typing is hidden — paste and press Enter){RESET}")
                value = getpass(f"   {field.key}{hint}: ").strip()
            else:
                value = input(f"   {field.key}{hint}: ").strip()
        except (EOFError, KeyboardInterrupt):
            say()
            err("Cancelled. Nothing was written.")
            sys.exit(130)

        if not value:
            value = existing

        if not value and field.required:
            err("This one is required — please paste a value.")
            continue

        if value and field.choices and value not in field.choices:
            warn(f"'{value}' isn't one of the suggested options. Using it anyway.")

        if value and field.validator:
            problem = field.validator(value)
            if problem:
                warn(problem)
                try:
                    again = input("   Use it anyway? [y/N]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    sys.exit(130)
                if again != "y":
                    continue

        return value


def is_configured(values: dict[str, str] | None = None) -> bool:
    values = read_env() if values is None else values
    return bool(values.get("OPENAI_API_KEY", "").strip())


def run_wizard(force: bool = False) -> bool:
    """Returns True if .env ends up configured."""
    say()
    say(f"{BOLD}{CYAN}🔐  SignalLab — Secrets Setup{RESET}")
    say(f"{DIM}    Your keys are stored locally in .env and never leave this machine.{RESET}")

    values = read_env()
    already = is_configured(values)

    if already and not force:
        rule("Existing configuration found")
        for field in FIELDS:
            shown = mask(values.get(field.key, "")) if field.secret else values.get(field.key, field.default)
            say(f"   {field.key:<22} {shown}")
        say()
        try:
            choice = input("   Keep this configuration? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            say()
            return True
        if choice in ("", "y", "yes"):
            ok("Keeping existing configuration.")
            return True

    rule("Enter your settings — press Enter to accept the [default]")
    for field in FIELDS:
        values[field.key] = ask(field, values.get(field.key, ""))

    for key, default, _ in ADVANCED:
        values.setdefault(key, default)

    rule("Verifying")
    verified, message = verify_openai_key(values["OPENAI_API_KEY"])
    if verified:
        ok(message)
    else:
        warn(message)
        try:
            cont = input("   Save anyway? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            sys.exit(130)
        if cont not in ("", "y", "yes"):
            err("Not saved. Re-run this script when you have a working key.")
            return False

    write_env(values)
    rule()
    ok(f"Saved → {ENV_PATH}")
    say(f"{DIM}   .env is listed in .gitignore, so it will not be committed.{RESET}")
    say(f"{DIM}   To change these later: run setup_secrets.py again (or edit .env).{RESET}")
    return True


# ── GUI front-end (tkinter pop-up) ───────────────────────────────────────────


def gui_available() -> tuple[bool, str]:
    """Can we realistically open a window here? Returns (yes, reason_if_not)."""
    if os.environ.get("SIGNALLAB_NO_GUI"):
        return False, "SIGNALLAB_NO_GUI is set"

    if not IS_WINDOWS and not IS_MACOS:
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return False, "no display detected (headless or SSH session)"

    try:
        import tkinter  # noqa: F401
    except Exception:
        hint = "tkinter is not installed"
        if not IS_WINDOWS and not IS_MACOS:
            hint += " — install it with: sudo apt install python3-tk"
        return False, hint

    return True, ""


def run_gui(force: bool = False) -> bool:
    """Show the setup pop-up. Returns True if .env was written (or already fine)."""
    import threading
    import tkinter as tk
    import webbrowser
    from tkinter import font as tkfont
    from tkinter import messagebox, ttk

    values = read_env()
    saved = {"ok": False}

    root = tk.Tk()
    root.title("SignalLab — Setup")
    root.resizable(False, False)
    try:
        root.call("tk", "scaling", 1.3)
    except Exception:
        pass

    base = tkfont.nametofont("TkDefaultFont")
    title_font = base.copy()
    title_font.configure(size=base.cget("size") + 6, weight="bold")
    label_font = base.copy()
    label_font.configure(weight="bold")
    small_font = base.copy()
    small_font.configure(size=max(base.cget("size") - 1, 8))

    style = ttk.Style(root)
    if "clam" in style.theme_names() and not (IS_WINDOWS or IS_MACOS):
        style.theme_use("clam")

    outer = ttk.Frame(root, padding=(28, 22, 28, 20))
    outer.grid(sticky="nsew")

    row = 0
    ttk.Label(outer, text="SignalLab Setup", font=title_font).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1
    ttk.Label(
        outer,
        text="One-time setup. Your key is stored locally in a file called .env\n"
             "and is never sent anywhere except OpenAI.",
        font=small_font,
        foreground="#555555",
        justify="left",
    ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(4, 14))
    row += 1
    ttk.Separator(outer, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 16))
    row += 1

    # ── API key ──
    ttk.Label(outer, text="OpenAI API key", font=label_font).grid(row=row, column=0, sticky="w")
    ttk.Label(outer, text="required", font=small_font, foreground="#b00020").grid(row=row, column=1, sticky="e")
    row += 1

    key_var = tk.StringVar(value=values.get("OPENAI_API_KEY", ""))
    key_entry = ttk.Entry(outer, textvariable=key_var, width=52, show="•")
    key_entry.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(4, 4))
    row += 1

    key_row = ttk.Frame(outer)
    key_row.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 14))
    show_var = tk.BooleanVar(value=False)

    def _toggle_show() -> None:
        key_entry.config(show="" if show_var.get() else "•")

    ttk.Checkbutton(key_row, text="Show key", variable=show_var, command=_toggle_show).pack(side="left")
    ttk.Button(
        key_row,
        text="Get a key from OpenAI ↗",
        command=lambda: webbrowser.open(KEY_SIGNUP_URL),
        width=24,
    ).pack(side="right")
    row += 1

    ttk.Label(
        outer,
        text="Paste the key that starts with “sk-”. You need a payment method on the\n"
             "OpenAI account. A typical SignalLab run costs about $0.10–$0.30.",
        font=small_font,
        foreground="#555555",
        justify="left",
    ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 16))
    row += 1

    # ── Models ──
    model_var = tk.StringVar(value=values.get("OPENAI_MODEL", "gpt-4o-mini"))
    embed_var = tk.StringVar(value=values.get("EMBED_MODEL", "text-embedding-3-small"))

    ttk.Label(outer, text="Reasoning model", font=label_font).grid(row=row, column=0, sticky="w")
    row += 1
    ttk.Combobox(
        outer, textvariable=model_var, values=["gpt-4o-mini", "gpt-4o"], state="readonly", width=30
    ).grid(row=row, column=0, columnspan=2, sticky="ew", pady=(4, 2))
    row += 1
    ttk.Label(
        outer,
        text="gpt-4o-mini is cheaper, faster, and has higher rate limits. Recommended.",
        font=small_font,
        foreground="#555555",
    ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 14))
    row += 1

    ttk.Label(outer, text="Embedding model", font=label_font).grid(row=row, column=0, sticky="w")
    row += 1
    ttk.Combobox(
        outer,
        textvariable=embed_var,
        values=["text-embedding-3-small", "text-embedding-3-large"],
        state="readonly",
        width=30,
    ).grid(row=row, column=0, columnspan=2, sticky="ew", pady=(4, 2))
    row += 1
    ttk.Label(
        outer,
        text="Leave as-is unless you know you want larger embeddings.",
        font=small_font,
        foreground="#555555",
    ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 12))
    row += 1

    # ── Status + buttons ──
    status_var = tk.StringVar(value="")
    status = ttk.Label(outer, textvariable=status_var, font=small_font, wraplength=430, justify="left")
    status.grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 10))
    row += 1

    buttons = ttk.Frame(outer)
    buttons.grid(row=row, column=0, columnspan=2, sticky="ew")
    buttons.columnconfigure(0, weight=1)

    def set_status(msg: str, colour: str = "#555555") -> None:
        status_var.set(msg)
        status.config(foreground=colour)

    def on_cancel() -> None:
        root.destroy()

    cancel_btn = ttk.Button(buttons, text="Cancel", command=on_cancel, width=12)
    cancel_btn.pack(side="right", padx=(8, 0))
    save_btn = ttk.Button(buttons, text="Save & Start", width=16)
    save_btn.pack(side="right")

    def finish(key: str) -> None:
        values["OPENAI_API_KEY"] = key
        values["OPENAI_MODEL"] = model_var.get()
        values["EMBED_MODEL"] = embed_var.get()
        for k, default, _ in ADVANCED:
            values.setdefault(k, default)
        write_env(values)
        saved["ok"] = True
        root.destroy()

    def on_save(_event=None) -> None:
        key = key_var.get().strip()

        if not key:
            set_status("Please paste your OpenAI API key to continue.", "#b00020")
            key_entry.focus_set()
            return

        problem = _validate_openai_key(key)
        if problem and not messagebox.askyesno("Double-check that key", f"{problem}\n\nUse it anyway?", parent=root):
            key_entry.focus_set()
            return

        save_btn.config(state="disabled")
        cancel_btn.config(state="disabled")
        set_status("Checking the key with OpenAI…")
        root.update_idletasks()

        holder: dict[str, tuple[bool, str]] = {}
        worker = threading.Thread(target=lambda: holder.update(res=verify_openai_key(key)), daemon=True)
        worker.start()

        def poll() -> None:
            if worker.is_alive():
                root.after(120, poll)
                return
            verified, message = holder.get("res", (False, "Verification skipped."))
            save_btn.config(state="normal")
            cancel_btn.config(state="normal")
            if verified:
                set_status(message, "#0a7a28")
                root.after(400, lambda: finish(key))
                return
            if messagebox.askyesno("Couldn't verify the key", f"{message}\n\nSave it anyway?", parent=root):
                finish(key)
            else:
                set_status(message, "#b00020")
                key_entry.focus_set()

        root.after(120, poll)

    save_btn.config(command=on_save)
    root.bind("<Return>", on_save)
    root.bind("<Escape>", lambda _e: on_cancel())
    root.protocol("WM_DELETE_WINDOW", on_cancel)

    if force and values.get("OPENAI_API_KEY"):
        set_status("Existing key loaded — edit it or just press Save & Start.")

    # Centre the window and make sure it comes to the front.
    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (w // 2)
    y = max((root.winfo_screenheight() // 3) - (h // 2), 0)
    root.geometry(f"+{x}+{y}")
    root.attributes("-topmost", True)
    root.lift()
    root.after(700, lambda: root.attributes("-topmost", False))
    key_entry.focus_force()

    root.mainloop()
    return saved["ok"]


# ── Front-end selection ──────────────────────────────────────────────────────


def configure(force: bool = False, prefer_gui: bool = True) -> bool:
    """Configure secrets using the best available front-end. Returns True on success."""
    if prefer_gui:
        can_gui, reason = gui_available()
        if can_gui:
            say()
            say(f"{CYAN}🔐  A setup window has opened.{RESET}")
            say(f"{DIM}    Can't see it? Check behind this window, or on your other screen.{RESET}")
            try:
                return run_gui(force=force)
            except Exception as e:  # noqa: BLE001 — any Tk failure falls back gracefully
                warn(f"Couldn't open the setup window ({e}). Falling back to this terminal.")
        else:
            say()
            say(f"{DIM}Setup window unavailable — {reason}. Using the terminal instead.{RESET}")

    return run_wizard(force=force)


def show_config() -> None:
    values = read_env()
    if not values:
        warn("No .env found. Run: python setup_secrets.py")
        return
    say(f"\n{BOLD}Current SignalLab configuration{RESET} {DIM}({ENV_PATH}){RESET}\n")
    for field in FIELDS:
        shown = mask(values.get(field.key, "")) if field.secret else values.get(field.key, "(not set)")
        say(f"   {field.key:<30} {shown}")
    for key, default, _ in ADVANCED:
        say(f"   {DIM}{key:<30} {values.get(key, default)}{RESET}")
    say()


def main() -> int:
    parser = argparse.ArgumentParser(description="Configure SignalLab API keys.")
    parser.add_argument("--check", action="store_true", help="Exit 0 if configured, 1 if not. No output.")
    parser.add_argument("--show", action="store_true", help="Print current config with the key masked.")
    parser.add_argument("--force", action="store_true", help="Re-prompt even if already configured.")
    parser.add_argument("--console", action="store_true", help="Force the terminal wizard, no pop-up.")
    parser.add_argument("--gui", action="store_true", help="Force the pop-up (fails loudly if unavailable).")
    args = parser.parse_args()

    if args.check:
        return 0 if is_configured() else 1
    if args.show:
        show_config()
        return 0

    if args.gui:
        can_gui, reason = gui_available()
        if not can_gui:
            err(f"Can't open a setup window here: {reason}")
            return 1
        return 0 if run_gui(force=args.force or True) else 1

    # Run standalone (not via start.py) → the user is asking to change settings.
    force = args.force or is_configured()
    return 0 if configure(force=force, prefer_gui=not args.console) else 1


if __name__ == "__main__":
    sys.exit(main())
