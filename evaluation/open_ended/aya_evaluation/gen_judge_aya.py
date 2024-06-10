import argparse
import re
import json
import openai
from openai.error import (
    RateLimitError,
    InvalidRequestError,
    Timeout,
    APIConnectionError,
    ServiceUnavailableError,
    APIError
)
import os
import time
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_answer", type=str, default=None, help="file name")
parser.add_argument("--judger", type=str, default='gpt-4')
parser.add_argument('--language_list', nargs='+', type=str, help='Language list')
args = parser.parse_args()

response_path = f"./model_answer/{args.model_answer}.json"
save_path = "./model_judgment/{judger}_{exp_name}.json".format(judger=args.judger, exp_name=args.model_answer)

# ============= Load the model outputs =============
with open(response_path) as f:
    model_outputs = json.load(f)

language_list = args.language_list

# ============= Load the judge template =============
template = "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. A good answer should follow these rules: \n1.It should be in the same language as the question. \n2. It should answer the request in the instruction. \n3.It should be factually and semantically comprehensible. \n4. It should be grammatically correct and fluent. \nBegin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".A human annotated answer is given for reference.\n\n[Question]\n{question}\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]\n\n[Reference]\n{reference}"

# # ============= Load the judge list if exists =============
# if os.path.exists(save_path):
#     with open(save_path, "r") as f:
#         judge_list = json.load(f)
# else:
#     judge_list = []
# existing_length = len(judge_list)
# print(f">> Existing length: {existing_length}")

def get_retry_time(err_info):
    z = re.search(r"after (\d+) seconds", err_info)
    if z:
        return int(z.group(1))
    return 1

def completion(messages, args):
    success = False
    while not success:
        try:
            response = openai.ChatCompletion.create(
                    model=args.judger,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=2048,
                )
            success = True
        except RateLimitError as e:
            print(e)
            retry_time = get_retry_time(str(e))
            time.sleep(retry_time)
        except Timeout as e:
            print(e)
            time.sleep(1)
        except APIConnectionError as e:
            print(e)
            time.sleep(1)
        except APIError as e:
            print(e)
            time.sleep(1)
        except ServiceUnavailableError as e:
            print(e)
            time.sleep(1)
        except InvalidRequestError as e:
            print(e)
            success = True
            response = {"choices": []}
        except Exception as e:
            print(e)
            success = True
            response = {"choices": []}
    return response


judge_dict = {}

for language in language_list:
    count = 0
    if language not in judge_dict:
        judge_dict[language] = []
    language_outputs = model_outputs[language]
    print("start judge language:",language)
    for output in tqdm(language_outputs):
        count += 1
        current_prompt = template.format(question=output["instruction"], answer=output["output"], reference=output["reference"])

        messages = [
            {"role": "system", "content": "You are a helpful assistant, that rates models by the quality of their answers."},
            {"role": "user", "content": f"{current_prompt}"}
        ]

        response = completion(messages, args)

        record_sample = {}
        record_sample["for_judge"] = current_prompt
        record_sample["response"] = response["choices"][0]["message"]["content"]

        judge_dict[language].append(record_sample)

        print("="*50, count, "="*50)
        print(record_sample["response"])

        with open(save_path, "w",encoding='utf-8') as f:
            json.dump(judge_dict, f, indent=4, ensure_ascii=False)