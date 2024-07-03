max_steps=160
num_rounds=100
checkpoint_step=50
batch_size=1
gradient_accumulation_steps=1
seq_length=1024
num_clients=100 
sample_clients=5 
lora_r=16
lora_alpha=32   # twice of lora_r
lr=2e-5

#dp parameters
dp_sigma=1e-3 #add this parameter below manually if you want to fix sigma
dp_delta=0.0001
dp_epsilon=1.0

local_data_dir=data/Fed-WildChat/single_turn/wildchat_100c_53k.json   # you may uncomment this line if your data is stored locally and include it in the python command
dataset_name=FedWildChatSingle
dataset_sample=53k
model_name_or_path=""
output_dir=./models/FedWildChat/single_turn_dp

gpu=0
fed_alg="feddp"


CUDA_VISIBLE_DEVICES=$gpu python -u main_sft.py \
 --dp_delta $dp_delta \
 --dp_epsilon $dp_epsilon \
 --local_data_dir $local_data_dir \
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
 --gradient_checkpointing \
 --template "alpaca" \
 --checkpoint_step $checkpoint_step
#  --dp_sigma $dp_sigma 
