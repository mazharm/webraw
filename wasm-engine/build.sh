#!/bin/bash
set -e
wasm-pack build --target web --out-dir ../frontend/src/wasm-pkg --release
