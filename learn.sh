#!/bin/sh
# Always import this worktree's tasks, including when reusing a sibling venv.
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
training_python=${TRAINING_PYTHON:-"$repo_dir/mjlab_microduck/.venv/bin/python"}
if [ ! -x "$training_python" ] && [ -z "${TRAINING_PYTHON:-}" ]; then
    training_python="$repo_dir/../Microduck-build-tutorial/mjlab_microduck/.venv/bin/python"
fi
if [ ! -x "$training_python" ]; then
    echo "Training Python not found: $training_python" >&2
    echo "Set TRAINING_PYTHON to an existing mjlab environment, or follow LEARNING.md." >&2
    exit 1
fi
export PYTHONPATH="$repo_dir/mjlab_microduck/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$training_python" "$repo_dir/mjlab_microduck/scripts/learning_pipeline.py" "$@"
