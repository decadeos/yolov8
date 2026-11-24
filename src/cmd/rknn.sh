#!/bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
EXP_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('${SCRIPT_DIR}/../../params.yaml'))['train']['name'])")

MODEL_ONNX="$1"
MODEL_ONNX=$(readlink -f "$MODEL_ONNX")
MODEL_DIR="$(dirname "$MODEL_ONNX")"

SAVE_DIR="${MODEL_DIR}/../output"

cd "$SCRIPT_DIR/../../../"
source .rknn_v/bin/activate
cd rknn-toolkit2/rknn_build
COPY_PATH="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

cp $MODEL_ONNX $COPY_PATH/onnx_models
ls -la
tree

python3 onnx2rknn.py
cp rknn_models/$EXP_NAME.rknn $SAVE_DIR

cd $SCRIPT_DIR/../../
ls -la $SAVE_DIR/*.rknn
source ../.venv/bin/activate