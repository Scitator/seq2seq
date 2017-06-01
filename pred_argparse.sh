#!/usr/bin/env bash

set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

model_dir=
data_dir=
dev_sources=
beam_width=
beam_k=

# Parse named args
while [ "$#" -gt 0 ]; do
    case "$1" in
        --model_dir)
            model_dir=$2
            shift
            shift
            ;;
        --data_dir)
            data_dir=$2
            shift
            shift
            ;;
        --dev_sources)
            dev_sources=$2
            shift
            shift
            ;;
        --beam_width)
            beam_width=$2
            shift
            shift
            ;;
        --beam_k)
            beam_k=$2
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
