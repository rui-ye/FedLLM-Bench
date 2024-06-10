gpus=(0 1 2 3)                    #a list of gpus to enable parallel generation on multiple gpus

lora_pathes=(
    
)                                           # a list of lora paths

for ((i=0; i<${#lora_pathes[@]}; i++)); do

    lora_path=${lora_pathes[$i]}

    gpu=${gpus[$i]}

    CUDA_VISIBLE_DEVICES=$gpu python gen_answer_aya.py \
    --base_model_path /GPFS/data/ruiye-1/models/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9 \
    --lora_path $lora_path \
    --language_list English Standard_Arabic Russian Simplified_Chinese Portuguese French Spanish Telugu &
done

wait