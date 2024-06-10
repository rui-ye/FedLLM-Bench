export OPENAI_API_KEY=                     # Your OpenAI api key here

NUM_CONCURRENT_API_CALL=1

model_answer_list=(

)

for model_answer in "${model_answer_list[@]}"; do
    python gen_judge_bench.py --judger gpt-4 --model_answer $model_answer --parallel $NUM_CONCURRENT_API_CALL
done