# Please change the name of the "ATTACK_METHOD" to eval your method!
# You can run this file directly!
ATTACK_METHOD=$1
SOURCE_MODEL=resnet18

if echo "$ATTACK_METHOD" | grep -q "ops"; then
  INPUT_DIR=../exp_data/exp_ops_overhead/${ATTACK_METHOD}/${SOURCE_MODEL}
else
  INPUT_DIR=../exp_data/exp_method_compare/${ATTACK_METHOD}/${SOURCE_MODEL}
fi

LABEL_FILE=../new_data/labels.csv
CHECKPOINT_PATH=./models/rs_imagenet/resnet50/noise_0.50/checkpoint.pth.tar
GPU_ID='0'

python rs/predict.py "${INPUT_DIR}" "${LABEL_FILE}" "${CHECKPOINT_PATH}" 0.50 prediction_outupt --alpha 0.001 --N 1000 --skip 100 --batch 1 --GPU_ID $GPU_ID # --targeted
