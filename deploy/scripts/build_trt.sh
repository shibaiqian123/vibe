#!/usr/bin/env bash
set -euo pipefail
DEPLOY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${VIBE_DEPLOY_BUILD:?Set VIBE_DEPLOY_BUILD to an independent build directory}"
cmake -S "$DEPLOY_ROOT/controller" -B "$VIBE_DEPLOY_BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$VIBE_DEPLOY_BUILD" -j 2
