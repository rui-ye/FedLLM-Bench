export OPENAI_API_KEY=                          # Your OpenAI api key here

model_answer_list=(

)
for model_answer in "${model_answer_list[@]}"; do
    python gen_judge_vicuna.py \
    --model_answer $model_answer \
    --judger gpt-4
done

# TODO: 
# OpenAI api key
# model_answer_list
