# Please change the name of the "ATTACK_METHOD" to eval your method!
# You can run this file directly!
# ATTACK_METHOD=fgsm
ATTACK_METHOD=$1
SOURCE_MODEL=resnet18

if echo "$ATTACK_METHOD" | grep -q "ops"; then
  INPUT_DIR=../../exp_data/exp_ops_overhead/${ATTACK_METHOD}/${SOURCE_MODEL}
else
  INPUT_DIR=../../exp_data/exp_method_compare/${ATTACK_METHOD}/${SOURCE_MODEL}
fi

OUTPUT_FILE=hgd_results/${ATTACK_METHOD}_hgd_results.txt
CHECKPOINT_DIR_PATH=../models
LABEL_FILE=../new_data/labels.csv
GPU_ID=0

cd hgd
python defense.py \
  --input_dir="${INPUT_DIR}" \
  --output_file="${OUTPUT_FILE}" \
  --checkpoint_dir_path="${CHECKPOINT_DIR_PATH}" \
  --GPU_ID="${GPU_ID}"

cd ..
python check_output.py \
  --output_file=hgd/"${OUTPUT_FILE}" \
  --label_file="${LABEL_FILE}"
# --targeted
