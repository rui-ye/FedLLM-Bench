import argparse
import re
import json
import ast
import numpy as np

one_score_pattern = re.compile("\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile("\[(\d+\.?\d*)\]")


def get_socres(file_path):
    print(file_path)
    with open(file_path, "r") as f:
        ori_judge = json.load(f)

    score_list = []
    
    for example in ori_judge:
        judgement = example["judgement"]
        match = re.search(one_score_pattern, judgement)
        if not match:
            match = re.search(one_score_pattern_backup, judgement)
        if match:
            score = ast.literal_eval(match.groups()[0])
        else:
            print("no score!!!")
            print(judgement)
            continue

        score_list.append(score)
                    
    
    print(f"Averaged score: {np.mean(score_list)}, std: {np.std(score_list)}")
    print("="*100)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_list",
        type=str,
        nargs="+",
        default=None,
        help="A list of judge results to be calculated",
    )
    args = parser.parse_args()

    for eval_name in args.eval_list:
        bench_name = eval_name.split("_")[0]
        file_path = f"./data/{bench_name}/model_judgment/gpt-4_{eval_name}.json"
        get_socres(file_path)
