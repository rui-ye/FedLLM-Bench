max_steps=5                           
num_rounds=90
checkpoint_step=30
batch_size=8
gradient_accumulation_steps=1
seq_length=512
num_clients=1
sample_clients=1
lr=1e-4

local_data_dir=data/Fed-ChatbotPA/local/local_154.json     # you may uncomment this line if your data is stored locally and include it in the python command
dataset_name=FedChatbotPA_local
dataset_sample=154
model_name_or_path=""
output_dir=./models/FedChatbotPA/local

gpu=0
fed_alg=local0

CUDA_VISIBLE_DEVICES=$gpu python main_dpo.py \
 --model_name_or_path $model_name_or_path \
 --dataset_name $dataset_name \
 --dataset_sample $dataset_sample \
 --fed_alg $fed_alg \
 --num_clients $num_clients \
 --sample_clients $sample_clients \
 --learning_rate $lr \
 --max_steps $max_steps \
 --num_rounds $num_rounds \
 --batch_size $batch_size \
 --gradient_accumulation_steps $gradient_accumulation_steps \
 --seq_length $seq_length \
 --use_peft \
 --load_in_8bit \
 --output_dir $output_dir \
 --template "alpaca" \
 --local_data_dir $local_data_dir \
 --checkpoint_step $checkpoint_step