#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for dir in reports logs; do
    target="$SCRIPT_DIR/$dir"
    if [ -d "$target" ]; then
        rm -rf "$target"/*
        echo "已清空: $target"
    fi
done

python3 "$SCRIPT_DIR/run_pipeline.py" --stage all --skip-data
