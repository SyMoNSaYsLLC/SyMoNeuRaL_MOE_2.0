#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Canonical install entry point. This script intentionally delegates to
# bootstrap_env.sh so first-time setup always happens in the correct order:
# venv -> pip upgrade -> requirements -> pip check -> audits.
exec "$ROOT/scripts/bootstrap_env.sh"
