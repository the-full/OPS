# Please change the name of the "attack" to eval your method!
# You can run this file directly!
# attack=fgsm
attack=$1
model=resnet18

if echo "$ATTACK_METHOD" | grep -q "ops"; then
  INPUT_DIR=../exp_data/exp_ops_overhead/${attack}/${model}
else
  INPUT_DIR=../exp_data/exp_method_compare/${attack}/${model}
fi

OUTPUT_DIR1=../defense/nrp/purified_data/${attack}/${model} #purified_data
OUTPUT_DIR2=defense/nrp/purified_data/${attack}/${model}    #purified_data
MODEL_PATH=../defense/models/NRP.pth
EVAL_DIR=new_data/
GPU_ID='0'

python nrp/purify.py --dir="${INPUT_DIR}" --output="${OUTPUT_DIR1}" --purifier NRP --model_pth ${MODEL_PATH} --dynamic --GPU_ID=${GPU_ID}

cd ..
python main.py --batchsize 10 --input_dir "${EVAL_DIR}" --output_dir "${OUTPUT_DIR2}" --eval --GPU_ID=${GPU_ID}

cd defense
