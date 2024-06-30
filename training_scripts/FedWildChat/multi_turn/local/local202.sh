max_steps=10
num_rounds=25
checkpoint_step=25
batch_size=4
gradient_accumulation_steps=4
seq_length=2048
num_clients=1
sample_clients=1
lora_r=16
lora_alpha=32   # twice of lora_r
lr=2e-5

local_data_dir="data/Fed-WildChat/multi_turn/local/local_202.json"       # you may uncomment this line if your data is stored locally and include it in the python command
dataset_name=FedWildChatMulti
dataset_sample=202
model_name_or_path="/GPFS/data/ruiye-1/models/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9"
output_dir=models/FedWildChat/multi_turn/local

gpu=0
fed_alg=local0

CUDA_VISIBLE_DEVICES=$gpu python main_sft.py \
 --learning_rate $lr \
 --model_name_or_path $model_name_or_path \
 --dataset_name $dataset_name \
 --dataset_sample $dataset_sample \
 --fed_alg $fed_alg \
 --num_clients $num_clients \
 --sample_clients $sample_clients \
 --max_steps $max_steps \
 --num_rounds $num_rounds \
 --batch_size $batch_size \
 --gradient_accumulation_steps $gradient_accumulation_steps \
 --seq_length $seq_length \
 --peft_lora_r $lora_r \
 --peft_lora_alpha $lora_alpha \
 --use_peft \
 --load_in_8bit \
 --output_dir $output_dir \
 --template "alpaca" \
 --local_data_dir $local_data_dir \
 --gradient_checkpointing \
 --checkpoint_step $checkpoint_step \
 --multi_turn_task
