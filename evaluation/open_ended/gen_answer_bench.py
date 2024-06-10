import datasets
import argparse
import json
import sys
sys.path.append("../../")
sys.path.append("../../../")
from tqdm import tqdm
import os
import torch
import sys

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.template import TEMPLATE_DICT

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--lora_path", type=str, default=None)
parser.add_argument("--template", type=str, default="alpaca")
parser.add_argument("--bench_name", type=str, default=None)
args = parser.parse_args()
print(args)

testset_path = os.path.join("./data",args.bench_name,'testset.json')

# ============= Load dataset =============
with open(testset_path,'r',encoding='utf-8') as file:
    data = json.load(file)


# ============= Extract model name from the path. The name is used for saving results. =============
if args.lora_path:
    pre_str, checkpoint_str = os.path.split(args.lora_path)
    _, exp_name = os.path.split(pre_str)
    checkpoint_id = checkpoint_str.split("-")[-1]
    model_name = f"{exp_name}_{checkpoint_id}"
else:
    pre_str, last_str = os.path.split(args.base_model_path)
    if last_str.startswith("full"):                 # if the model is merged as full model
        _, exp_name = os.path.split(pre_str)
        checkpoint_id = last_str.split("-")[-1]
        model_name = f"{exp_name}_{checkpoint_id}"
    else:
        model_name = last_str                       # mainly for base model
        exp_name = model_name


# ============= Generate responses =============
device = 'cuda'
model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.float16).to(device)
if args.lora_path is not None:
    model = PeftModel.from_pretrained(model, args.lora_path, torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)

result_path = f"./data/{args.bench_name}/model_answer/{model_name}.json"

print(result_path)

answer_list = []

template = TEMPLATE_DICT[args.template][0]

print(f">> You are using template: {template}")

for example in tqdm(data):

    instruction = template.format(example["instruction"], "", "")[:-1]

    input_ids = tokenizer.encode(instruction, return_tensors="pt").to(device)
    output_ids = model.generate(inputs=input_ids, max_new_tokens=1024, do_sample=True, top_p=1.0, temperature=0.7)
    output_ids = output_ids[0][len(input_ids[0]):]
    result = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    example["response"] = result

    print(f"\nInput: \n{instruction}")
    print(f"\nOutput: \n{result}")
    print("="*100)

    answer_list.append(example)

    with open(result_path,'w',encoding='utf-8') as output:
        json.dump(answer_list,output,indent=4,ensure_ascii=False)


