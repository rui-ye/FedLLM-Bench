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
parser.add_argument('--language_list', nargs='+', type=str, help='Language list')
args = parser.parse_args()
print(args)

lang_template_dict = {
    'Standard_Arabic':'arabic_alpaca',
    'English':'alpaca',
    'Simplified_Chinese':'chinese_alpaca',
    'Portuguese':'portuguese_alpaca',
    'Telugu':'telugu_alpaca',
    'Russian':'russian_alpaca',
    'French':'french_alpaca',
    'Spanish':'spanish_alpaca'
}


# ============= Load dataset =============
with open('test_datasets.json','r') as  file:
    data = json.load(file)

# language_list = ["Standard Arabic","English","Simplified Chinese","Portuguese","Telugu"]
language_list = args.language_list


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

result_path = f"./model_answer/{model_name}.json"

print(result_path)

answer_dict = {}
for language in language_list:
    if language not in answer_dict:
        answer_dict[language] = []

    eval_dataset = data[language]
    template = TEMPLATE_DICT['alpaca'][0]
    print(f">> You are using template: {template}")
    for example in tqdm(eval_dataset, desc="Generating ansewr", unit="example"):
        instruction = template.format(example["instruction"], "", "")[:-1]
    
        input_ids = tokenizer.encode(instruction, return_tensors="pt").to(device)
        output_ids = model.generate(inputs=input_ids, max_new_tokens=1024, do_sample=True, top_p=1.0, temperature=0.7)
        output_ids = output_ids[0][len(input_ids[0]):]
        result = tokenizer.decode(output_ids, skip_special_tokens=True)
        example['output'] = result
        example['generator'] = exp_name

        answer_dict[language].append(example)
        print(f"\nInput: \n{instruction}")
        print(f"\nOutput: \n{result}")
        print("="*100)

        with open(result_path,'w',encoding='utf-8') as output:
            json.dump(answer_dict,output,indent=4,ensure_ascii=False)


