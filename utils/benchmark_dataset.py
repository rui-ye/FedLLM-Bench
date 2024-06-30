import json
from datasets import load_dataset, Dataset
from data_module import make_supervised_data_module

def get_local_datasets(file_path):
    dataset = load_dataset("json", data_files=file_path)['train']
    return [dataset]

def get_fed_datasets(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        data = json.load(f)

    local_datasets = []
    for value in data.values():
        dataset = Dataset.from_list(value)
        local_datasets.append(dataset)
    
    return local_datasets

def get_multi_turn_dataset(fed_alg, data_dir, tokenizer):
    data = json.load(open(data_dir,'r',encoding='utf-8'))
    data_collator_list = []
    if fed_alg.startswith('local'):
        data_collator_list = [make_supervised_data_module(tokenizer,data)]
        sample_num_list = [len(data_collator_list[0])]
    else:
        for value in data.values():
            data_collator_list.append(make_supervised_data_module(tokenizer, value))
        sample_num_list = [len(elem) for elem in data_collator_list]
    
    return data_collator_list, sample_num_list



