export OPENAI_API_KEY=             # Your OpenAI api key here

model_answer_list=(
    
)

for model_answer in "${model_answer_list[@]}"; do
    python gen_judge_aya.py --judger gpt-4 --model_answer $model_answer --language_list Spanish
done