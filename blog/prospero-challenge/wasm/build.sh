#!/bin/sh
set -eu

ROOT_DIR=$(cd "$(dirname "$0")/../../.." && pwd)
EMSDK_DIR="$ROOT_DIR/../emsdk"
SOURCE_DIR="$ROOT_DIR/../prospero_backward"
OUT_DIR="$ROOT_DIR/blog/prospero-challenge/wasm"
OMP_OUT_DIR="$ROOT_DIR/blog/prospero-challenge/wasm-omp"
OPENMP_WASM_PREFIX="$ROOT_DIR/../openmp-wasm-install"

export EMSDK_QUIET=1
. "$EMSDK_DIR/emsdk_env.sh" >/dev/null

emcc \
  -O3 \
  -flto \
  -DNDEBUG \
  -ffast-math \
  -ffp-contract=fast \
  -msimd128 \
  -I"$SOURCE_DIR" \
  "$SOURCE_DIR/vm.c" \
  "$ROOT_DIR/blog/prospero-challenge/wasm_benchmark.c" \
  -lm \
  -sMODULARIZE=1 \
  -sEXPORT_NAME=createProsperoModule \
  -sENVIRONMENT=web \
  -sEXPORTED_FUNCTIONS='["_malloc","_free"]' \
  -sEXPORTED_RUNTIME_METHODS='["ccall","cwrap","stringToUTF8","lengthBytesUTF8","HEAPU8"]' \
  -sALLOW_MEMORY_GROWTH=1 \
  -sINITIAL_MEMORY=67108864 \
  -sSTACK_SIZE=8388608 \
  -o "$OUT_DIR/prospero.js"

node -e "const fs=require('fs'); const out=process.argv[1]; const b=fs.readFileSync(out + '/prospero.wasm').toString('base64'); fs.writeFileSync(out + '/prospero-binary.js','window.PROSPERO_WASM_BASE64 = '+JSON.stringify(b)+';\\n');" "$OUT_DIR"

mkdir -p "$OMP_OUT_DIR"

emcc \
  -O3 \
  -flto \
  -DNDEBUG \
  -ffast-math \
  -ffp-contract=fast \
  -fopenmp \
  -pthread \
  -msimd128 \
  -I"$SOURCE_DIR" \
  -I"$OPENMP_WASM_PREFIX/include" \
  "$SOURCE_DIR/vm.c" \
  "$ROOT_DIR/blog/prospero-challenge/wasm_benchmark.c" \
  "$OPENMP_WASM_PREFIX/lib/libomp.a" \
  -lm \
  -sMODULARIZE=1 \
  -sEXPORT_NAME=createProsperoOmpModule \
  -sENVIRONMENT=web,worker \
  -sEXPORTED_FUNCTIONS='["_malloc","_free"]' \
  -sEXPORTED_RUNTIME_METHODS='["ccall","cwrap","stringToUTF8","lengthBytesUTF8","HEAPU8"]' \
  -sPTHREAD_POOL_SIZE=8 \
  -sINITIAL_MEMORY=268435456 \
  -sSTACK_SIZE=8388608 \
  -sDEFAULT_PTHREAD_STACK_SIZE=8388608 \
  -o "$OMP_OUT_DIR/prospero.js"

node -e "const fs=require('fs'); const out=process.argv[1]; const b=fs.readFileSync(out + '/prospero.wasm').toString('base64'); fs.writeFileSync(out + '/prospero-binary.js','window.PROSPERO_OMP_WASM_BASE64 = '+JSON.stringify(b)+';\\n');" "$OMP_OUT_DIR"
