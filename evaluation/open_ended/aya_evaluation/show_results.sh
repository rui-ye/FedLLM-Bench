model_answer_path=(
    aya_666_local0_c1s1_i10_b4a4_l1024_r16a32_20240526103941_40
)

# Constructing the eval_list string
eval_list="--eval_list"
for model in "${model_answer_path[@]}"; do
    eval_list+=" $model"
done

# Running show_results.py with eval_list
python show_results.py $eval_list