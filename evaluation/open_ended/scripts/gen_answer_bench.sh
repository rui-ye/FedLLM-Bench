gpus=()

lora_pathes=(
    
)   

bench_name=
for ((i=0; i<${#lora_pathes[@]}; i++)); do

    lora_path=${lora_pathes[$i]}

    gpu=${gpus[$i]}
    
    # 启动 gen_answer.py 进程
    CUDA_VISIBLE_DEVICES=$gpu python gen_answer_bench.py \
    --base_model_path  \
    --lora_path $lora_path \
    --template alpaca \
    --bench_name $bench_name &
done

wait
