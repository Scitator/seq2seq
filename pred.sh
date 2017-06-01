#!/usr/bin/env bash

set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=${DIR}

source ${DIR}/pred_argparse.sh

export LD_LIBRARY_PATH=/usr/local/cuda/extra/CUPTI/lib64:$LD_LIBRARY_PATH

export MODEL_DIR=$model_dir
export DATA_DIR=$data_dir

export PRED_DIR=${MODEL_DIR}/pred
mkdir -p ${PRED_DIR}

export DEV_SOURCES=$dev_sources

python -m ${DIR}/bin.infer \
  --tasks "
    - class: DecodeText
      params:
        unk_replace: False" \
  --model_dir $MODEL_DIR \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_delimiter: ' '
      source_files:
        - $DEV_SOURCES" \
  >  ${PRED_DIR}/predictions.txt
