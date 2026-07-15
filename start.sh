#!/usr/bin/env bash
# ===========================================================
#  SignalLab - macOS / Linux launcher
#  Double-click (macOS: use start.command) or run: ./start.sh
# ===========================================================

cd "$(dirname "$0")" || exit 1

PY=""
for candidate in python3.12 python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' >/dev/null 2>&1; then
            PY="$candidate"
            break
        fi
    fi
done

if [ -z "$PY" ]; then
    echo
    echo "-----------------------------------------------------------"
    echo " Python 3.10 or newer was not found on this computer."
    echo
    if [ "$(uname)" = "Darwin" ]; then
        echo " Install it with either:"
        echo "   • Homebrew:  brew install python@3.12"
        echo "   • Download:  https://www.python.org/downloads/"
    else
        echo " Install it with your package manager, for example:"
        echo "   • Ubuntu/Debian:  sudo apt install python3 python3-venv"
        echo "   • Fedora:         sudo dnf install python3"
    fi
    echo
    echo " Then run this file again."
    echo "-----------------------------------------------------------"
    echo
    read -r -p "Press Enter to close..." _
    exit 1
fi

"$PY" start.py "$@"
RC=$?

if [ $RC -ne 0 ] && [ $RC -ne 130 ]; then
    echo
    echo "-----------------------------------------------------------"
    echo " SignalLab stopped with an error. The message above explains why."
    echo "-----------------------------------------------------------"
    read -r -p "Press Enter to close..." _
fi

exit $RC
