gpus=()

lora_pathes=(
    
)
for ((i=0; i<${#lora_pathes[@]}; i++)); do

    lora_path=${lora_pathes[$i]}

    gpu=${gpus[$i]}
    
    CUDA_VISIBLE_DEVICES=$gpu python gen_model_answer_mt.py \
    --base_model_path  \
    --lora_path $lora_path \
    --template alpaca &
done

wait
