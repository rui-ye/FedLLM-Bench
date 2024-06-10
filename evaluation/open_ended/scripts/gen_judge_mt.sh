export OPENAI_API_KEY=                              # Your OpenAI api key here
NUM_CONCURRENT_API_CALL=1

python gen_judge_mtbench.py --judge_model gpt-4-1106-preview --model_list  --parallel $NUM_CONCURRENT_API_CALL

# TODO: 
# OpenAI api key
# --model_list

