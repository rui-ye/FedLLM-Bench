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
args = parser.parse_args()

bench_name = args.model_answer.split("_")[0]

response_path = f"./data/{bench_name}/model_answer/{args.model_answer}.json"
save_path = f"./data/{bench_name}/model_judgment/{args.judger}_{args.model_answer}.json"

# ============= Load the model outputs =============
with open(response_path) as f:
    model_outputs = json.load(f)

# ============= Load the judge template =============
template = "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user's instruction displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\". \n\n[Question]\n{question}\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]"

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


judges = []

for example in model_outputs:

    prompt = template.format(question=example["instruction"],answer=example["response"])
    print(prompt)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant, that rates models by the quality of their answers."},
        {"role": "user", "content": f"{prompt}"}
    ]

    response = completion(messages, args)

    judgement = response["choices"][0]["message"]["content"]

    # example["judgement"] = judgement
    example["judgement"] = judgement
    judges.append(example)

    print("="*100)
    print(judgement)

    with open(save_path, "w",encoding='utf-8') as output:
        json.dump(judges, output, indent=4, ensure_ascii=False)