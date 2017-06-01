#!/usr/bin/env bash

set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=${DIR}

source ${DIR}/argparse.sh

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
        unk_replace: False
    - class: DumpBeams
      params:
        file: ${PRED_DIR}/beams.npz" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: ${beam_width}" \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_delimiter: ' '
      source_files:
        - $DEV_SOURCES" \
  >  ${PRED_DIR}/beam_predictions.txt


python ${DIR}/bin/tools/generate_beam_topk.py \
  -d ${PRED_DIR}/beams.npz \
  -v ${DATA_DIR}/vocab.txt \
  -s " " \
  -k $beam_k > ${PRED_DIR}/beam_predictions_top.txt

cat ${PRED_DIR}/beam_predictions_top.txt |\
   awk -F '\t' '{print $2}' > ${PRED_DIR}/beam_predictions_top_clear_tmp.txt


export DEV_SOURCES2="${PRED_DIR}/beam_predictions_top_clear_tmp.txt"

python -m ${DIR}/bin.infer \
  --tasks "
    - class: DecodeText
      params:
        unk_replace: False
    - class: DumpBeams
      params:
        file: ${PRED_DIR}/beams.npz" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 30" \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_delimiter: ' '
      source_files:
        - $DEV_SOURCES2" \
  >  ${PRED_DIR}/beam_predictions.txt


python ${DIR}/bin/tools/generate_beam_topk.py \
  -d ${PRED_DIR}/beams.npz \
  -v ${DATA_DIR}/vocab.txt \
  -s " " \
  -k $beam_k > ${PRED_DIR}/beam_predictions_top.txt

cat ${PRED_DIR}/beam_predictions_top.txt |\
   awk -F '\t' '{print $2}' > ${PRED_DIR}/beam_predictions_top_clear.txt

${DIR}/bin/tools/merge_beam_source_targets.py \
    --input ${DEV_SOURCES} ${DEV_SOURCES2} ${PRED_DIR}/beam_predictions_top_clear.txt \
    > ${PRED_DIR}/beam_reversed_top.txt
