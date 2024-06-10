model_answer_list=(
    
)

# Constructing the eval_list string
eval_list="--eval_list"
for model in "${model_answer_list[@]}"; do
    eval_list+=" $model"
done

# Running show_results.py with eval_list
python show_results_bench.py $eval_list

# TODO: 
# model_answer_list