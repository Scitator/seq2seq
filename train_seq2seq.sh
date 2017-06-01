#!/usr/bin/env bash

set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

data_path=
model_config=
train_config=
model_dir=
train_steps=1000000
batch_size=128
eval_every=100000

# Parse named args
while [ "$#" -gt 0 ]; do
    case "$1" in
        --data_path)
            data_path=$2
            shift
            shift
            ;;
        --model_config)
            model_config=$2
            shift
            shift
            ;;
        --train_config)
            train_config=$2
            shift
            shift
            ;;
        --model_dir)
            model_dir=$2
            shift
            shift
            ;;
        --train_steps)
            train_steps=$2
            shift
            shift
            ;;
        --batch_size)
            batch_size=$2
            shift
            shift
            ;;
        --eval_every)
            eval_every=$2
            shift
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done


export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

export VOCAB_SOURCE="$data_path/vocab.txt"
export VOCAB_TARGET="$data_path/vocab.txt"

export TRAIN_SOURCES="$data_path/train_sources.txt"
export TRAIN_TARGETS="$data_path/train_targets.txt"

export DEV_SOURCES="$data_path/test_sources.txt"
export DEV_TARGETS="$data_path/test_targets.txt"
export DEV_TARGETS_REF="$data_path/test_targets.txt"

export TRAIN_STEPS=$train_steps

export MODEL_DIR=$model_dir
mkdir -p $MODEL_DIR

python -m bin.train \
  --config_paths="
      ${model_config},
      ${train_config},
      ./example_configs/text_metrics_raw_bleu.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_delimiter: ' '
      target_delimiter: ' '
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_delimiter: ' '
       target_delimiter: ' '
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size $batch_size \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR \
  --eval_every_n_steps=$eval_every \
  --keep_checkpoint_max=50
