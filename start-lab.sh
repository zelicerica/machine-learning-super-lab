#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "$0")"
if [ -n "${SML_LAB_PYTHON:-}" ]; then
  lab_python="$SML_LAB_PYTHON"
elif [ -x ".venv/bin/python" ]; then
  lab_python=".venv/bin/python"
elif command -v python3.12 >/dev/null 2>&1; then
  lab_python="python3.12"
elif command -v python3 >/dev/null 2>&1; then
  lab_python="python3"
else
  echo "Install Python 3.12, then run this launcher again. See README.md."
  exit 1
fi
if ! "$lab_python" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3,12) else 1)'; then
  echo "The selected interpreter must be Python 3.12. Check SML_LAB_PYTHON or your .venv."
  exit 1
fi
exec "$lab_python" server.py "$@"
