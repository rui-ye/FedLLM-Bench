gpus=()

lora_pathes=(
    
)
for ((i=0; i<${#lora_pathes[@]}; i++)); do

    lora_path=${lora_pathes[$i]}

    gpu=${gpus[$i]}

    CUDA_VISIBLE_DEVICES=$gpu python gen_model_answer.py \
    --base_model_path  \
    --lora_path $lora_path \
    --template alpaca \
    --bench_name advbench &
done

wait
# CUDA_VISIBLE_DEVICES=$gpu python gen_model_answer.py \
#  --base_model_path /GPFS/data/xhpang-1/LLM/alpaca_recovered \
#  --template alpaca \
#  --bench_name advbench \
#  --lora_path /GPFS/data/ruige-1/OpenFedLLMBenchmark/dpo_models/chatbot/chatbot_76_local0_c1s1_i5_b8a1_l512_r8a16_20240522173530/checkpoint-20
 