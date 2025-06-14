# Please change the name of the "ATTACK_METHOD" to eval your method!
# You can run this file directly!
ATTACK_METHOD=$1
SOURCE_MODEL=resnet18

if echo "$ATTACK_METHOD" | grep -q "ops"; then
  INPUT_DIR=../../exp_data/exp_ops_overhead/${ATTACK_METHOD}/${SOURCE_MODEL}
else
  INPUT_DIR=../../exp_data/exp_method_compare/${ATTACK_METHOD}/${SOURCE_MODEL}
fi

OUTPUT_FILE=at_results/${ATTACK_METHOD}_${SOURCE_MODEL}.txt
CHECKPOINT_DIR_PATH=../models
LABEL_FILE=../new_data/labels.csv
GPU_ID='0'

cd at
# CUDA_VISIBLE_DEVICES=0 \
python main_fast.py "${INPUT_DIR}" --config configs/configs_fast_4px_evaluate.yml --output_prefix "${OUTPUT_FILE}" \
  --resume "${CHECKPOINT_DIR_PATH}/imagenet_model_weights_4px.pth.tar" \
  --evaluate --restarts 10 --GPU_ID $GPU_ID

cd ..
python check_output.py \
  --output_file=at/"${OUTPUT_FILE}" \
  --label_file="${LABEL_FILE}"
# --targeted
