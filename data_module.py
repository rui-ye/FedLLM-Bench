from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from functools import cache
import os
from typing import Dict, List, Optional
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    T5Tokenizer,
)
import warnings


from conversation import Conversation, get_conv_template, SeparatorStyle
from compression import load_compress_model

## start at fuction: make_supervised_data_module

IGNORE_TOKEN_ID = LabelSmoother.ignore_index # -100

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )

class BaseModelAdapter:
    """The base and the default model adapter."""

    use_fast_tokenizer = True

    def match(self, model_path: str):
        return True

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=self.use_fast_tokenizer,
                revision=revision,
                trust_remote_code=True,
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=False, revision=revision, trust_remote_code=True
            )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                **from_pretrained_kwargs,
            )
        except NameError:
            model = AutoModel.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                **from_pretrained_kwargs,
            )
        return model, tokenizer

    def load_compress_model(self, model_path, device, torch_dtype, revision="main"):
        return load_compress_model(
            model_path,
            device,
            torch_dtype,
            use_fast=self.use_fast_tokenizer,
            revision=revision,
        )

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("one_shot")

def remove_parent_directory_name(model_path):
    """Remove parent directory name."""
    if model_path[-1] == "/":
        model_path = model_path[:-1]
    return model_path.split("/")[-1]

class VicunaAdapter(BaseModelAdapter):
    "Model adapter for Vicuna models (e.g., lmsys/vicuna-7b-v1.5)" ""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        return "vicuna" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        revision = from_pretrained_kwargs.get("revision", "main")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=self.use_fast_tokenizer, revision=revision
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        self.raise_warning_for_old_weights(model)
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        if "v0" in remove_parent_directory_name(model_path):
            return get_conv_template("one_shot")
        return get_conv_template("vicuna_v1.1")

    def raise_warning_for_old_weights(self, model):
        if isinstance(model, LlamaForCausalLM) and model.model.vocab_size > 32000:
            warnings.warn(
                "\nYou are probably using the old Vicuna-v0 model, "
                "which will generate unexpected results with the "
                "current fastchat.\nYou can try one of the following methods:\n"
                "1. Upgrade your weights to the new Vicuna-v1.3: https://github.com/lm-sys/FastChat#vicuna-weights.\n"
                "2. Use the old conversation template by `python3 -m fastchat.serve.cli --model-path /path/to/vicuna-v0 --conv-template one_shot`\n"
                "3. Downgrade fschat to fschat==0.1.10 (Not recommended).\n"
            )

class StableVicunaAdapter(BaseModelAdapter):
    """The model adapter for StableVicuna"""

    def match(self, model_path: str):
        return "stable-vicuna" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("stable-vicuna")

class OasstLLaMAAdapter(BaseModelAdapter):
    """The model adapter for OpenAssistant/oasst-sft-7-llama-30b"""

    use_fast_tokenizer = False

    def match(self, model_path: str):
        model_path = model_path.lower()
        if "openassistant-sft-7-llama-30b-hf" in model_path:
            return True
        return "oasst" in model_path and "pythia" not in model_path

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("oasst_llama")

class CodeLlamaAdapter(BaseModelAdapter):
    """The model adapter for CodeLlama (e.g., codellama/CodeLlama-34b-hf)"""

    def match(self, model_path: str):
        return "codellama" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("llama-2")

class PhindCodeLlamaAdapter(CodeLlamaAdapter):
    """The model adapter for Phind-CodeLlama (e.g., Phind/Phind-CodeLlama-34B-v2)"""

    def match(self, model_path: str):
        return "phind-codellama-" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("phind")
    
class Llama2Adapter(BaseModelAdapter):
    """The model adapter for Llama-2 (e.g., meta-llama/Llama-2-7b-hf)"""

    def match(self, model_path: str):
        return "llama-2" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().load_model(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("llama-2")
    
class Llama2ChangAdapter(Llama2Adapter):
    """The model adapter for Llama2-ko-chang (e.g., lcw99/llama2-ko-chang-instruct-chat)"""

    def match(self, model_path: str):
        return "llama2-ko-chang" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("polyglot_changgpt")


class TinyLlamaAdapter(BaseModelAdapter):
    """The model adapter for TinyLlama (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0)"""

    def match(self, model_path: str):
        return "tinyllama" in model_path.lower()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("TinyLlama")

def get_conversation_template(model_path: str) -> Conversation:
    """Get the default conversation template."""
    adapter = get_model_adapter(model_path)
    return adapter.get_default_conv_template(model_path)

model_adapters: List[BaseModelAdapter] = []

def register_model_adapter(cls):
    """Register a model adapter."""
    model_adapters.append(cls())

# Note: the registration order matters.
# The one registered earlier has a higher matching priority.
register_model_adapter(StableVicunaAdapter)
register_model_adapter(VicunaAdapter)
register_model_adapter(OasstLLaMAAdapter)
register_model_adapter(PhindCodeLlamaAdapter)
register_model_adapter(CodeLlamaAdapter)
register_model_adapter(TinyLlamaAdapter)
register_model_adapter(Llama2Adapter)

@cache
def get_model_adapter(model_path: str) -> BaseModelAdapter:
    """Get a model adapter for a model_path."""
    model_path_basename = os.path.basename(os.path.normpath(model_path))

    # Try the basename of model_path at first
    for adapter in model_adapters:
        if adapter.match(model_path_basename) and type(adapter) != BaseModelAdapter:
            return adapter

    # Then try the full path
    for adapter in model_adapters:
        if adapter.match(model_path):
            return adapter

    raise ValueError(f"No valid model adapter for {model_path}")

local_rank = 0
def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, train_json, eval_data_path=None
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        SupervisedDataset
    )
    rank0_print("Loading data...")

    # train_json = json.load(open(data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)

    if eval_data_path:
        eval_json = json.load(open(eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    else:
        eval_dataset = None


    # print('train_dataset',train_dataset)
    # return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)
    return train_dataset



def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = get_conversation_template("vicuna")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # conversations = [s.replace('</s>USER', '</s> USER') if '</s>USER' in s else s for s in conversations]
    # print('conversation length',len(conversations))
    count = 0
    for idx in range(len(conversations)):
        # if idx < 10 : # print examples
        # print(conversations[idx]) 
        if "</s>USER" in conversations[idx]:
            count = count + 1
    print('total count:',count)
    # print('tokenizer.model_max_length:',tokenizer.model_max_length)

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        
        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            if (not ('ASSISTANT:' in turn)): # user ends the conversation, set cur_len to total_len manually
                cur_len = total_len 
            turn_len = len(tokenizer(turn).input_ids)
            # turn_len = len(tokenizer(turn, add_special_tokens=False).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
            if i == 0:
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            else:
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID

            if i == 0 :
                cur_len += turn_len
            else:
                cur_len += (turn_len + 1) 


            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if True:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            # rank0_print(tokenizer.decode(z))
            # exit()
        

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


    
