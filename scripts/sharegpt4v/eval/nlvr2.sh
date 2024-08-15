#!/bin/bash

##### For slurm evaluation
# srun -p Your partion --gres gpu:8 bash scripts/sharegpt4v/eval/gqa.sh
##### For single node evaluation, you can vary the gpu numbers.
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/sharegpt4v/eval/gqa.sh

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT=${1-"share4v-7b"}
SPLIT="share4v_nlvr2_testdev_balanced"
NLVR2DIR="./playground/data/eval/nlvr2/"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m share4v.eval.model_vqa_nlvr \
        --model-path Lin-Chen/ShareGPT4V-7B \
        --question-file ./playground/data/eval/nlvr2/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/nlvr2/ \
        --answers-file ./playground/data/eval/nlvr2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/nlvr2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/nlvr2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_nlvr2_for_eval.py --src $output_file --dst $NLVR2DIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced
