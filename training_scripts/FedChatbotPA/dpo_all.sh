gpu=0
max_steps=10
num_rounds=200
batch_size=1
gradient_accumulation_steps=8
seq_length=512
num_clients=747
sample_clients=10
lr=1e-4

local_data_dir=benchmarkdata/chatbot-arena/chatbot_by_client-10k.json     # you may uncomment this line if your data is stored locally and include it in the python command
dataset_name=chatbot_dpo
# dataset_name="HuggingFaceH4/ultrafeedback_binarized"
dataset_sample=9508
model_name_or_path="/GPFS/data/xhpang-1/LLM/alpaca_recovered"
output_dir=./models/FedChatbotPA

fed_alg=scaffold
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
 --dynamic_local_step