import json
import requests
import time
from tqdm import tqdm
import os
import sys
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

model_list = ['gpt-4o-mini-2024-07-18', 'gpt-4o-2024-08-06', 'gpt-4o-2024-11-20', 'o1']
best_model = 'gpt-4o-2024-11-20'
ci_best_model = 'gpt-4o-2024-08-06'
back_model = 'gpt-4o-mini-2024-07-18'

last_sentence = ''

def requestGPT4(instruction, query, ak, temperature, model, max_retries=100000000):
    headers = {"Content-Type": "application/json"}
    data = {
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": query}
        ],
        "model": model, 
        "max_tokens": 1000,
        "temperature": temperature,
        "top_p": 0,
        "logit_bias": {},
        "n": 1,
        "stream": False
    }

    for attempt in range(max_retries):
        try:
            # 发送请求 TODO 这里需要替换成你能用的接口url
            response = requests.post(f'https://***/?ak={ak}', headers=headers, json=data)
            res = json.loads(response.text)
            
            result = res['choices'][0]['message']['content'].strip()
            return result  

        except Exception as e:
            time.sleep(2) 
    return f"Failed to get a response after {max_retries} retries."

# 请求豆包
def requestDoubao(instruction, query, max_retries=100000000):
    client = OpenAI(
        # TODO 这里需要替换成你能用的接口url
        base_url="****",
        api_key="****",
    )
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(

                model="ep-20250703174207-j75k9",
                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": query},
                ],
            )
            res=json.loads(response.model_dump_json())
            results,thinking_res=res["choices"][0]["message"]["content"],res["choices"][0]["message"]["reasoning_content"]
            return results
        except Exception as e:
            time.sleep(2)  
    return f"Failed to get a response after {max_retries} retries."

# 请求deepseek-r1
def requestDeepseek(instruction, query, max_retries=100000000):
    client = OpenAI(
        # TODO 同上替换
        api_key = "****",
        base_url = "****",
    )
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
            model = "ep-20250312204235-fc5xn", 
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": query},
            ],
        )
            res=completion.choices[0].message.content.strip()
            results,thinking_res=res["choices"][0]["message"]["content"],res["choices"][0]["message"]["reasoning_content"]
            return results 
        except Exception as e:
            time.sleep(2)  
    return f"Failed to get a response after {max_retries} retries."
   

import requests
import json
import random
import time

def requestGPT4_plus(instruction, query, label, ak, temperature, model='gpt-4o-mini-2024-07-18', max_retries=100000000):
    headers = {"Content-Type": "application/json"}
    data = {
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": query}
        ],
        "model": model, 
        "max_tokens": 1000,
        "top_p": 0,
        "logit_bias": {},
        "n": 1,
        "stream": False
    }

    for attempt in range(max_retries):
        try:
            data["temperature"] = temperature
            response = requests.post(f'https://search.bytedance.net/gpt/openapi/online/v2/crawl?ak={ak}', headers=headers, json=data)
            res = json.loads(response.text)
            result = res['choices'][0]['message']['content'].strip()
            if any(char in result[-25:] for char in label):
                return result
            temperature = round(random.uniform(0.1, 1.5), 2) 
            print(f"[INFO] Adjusting temperature to {temperature} and retrying request...")

        except Exception as e:
            print(f"[ERROR] Request failed on attempt {attempt + 1}: {e}")
            time.sleep(5)  

    return f"Failed to get a response after {max_retries} retries."

def process_manyidu_v3(temp, instruction, ak, model):
    temp = json.loads(temp)
    cur_input = temp['prompt'] + '\nOnly tell me the final answer.\n'
    doubao_out = requestDoubao(instruction, cur_input)
    return doubao_out


def manyidu_pipeline(model, instruction, file_path):
    ll = []
    with open(file_path,"r",encoding='utf-8') as f:
        for line in f:
            ll.append(line)
    ak = "Yr7pC78Eo1wFxcxqHJ5gqIGjQ0sW9LBp"
    output_path = file_path.replace('.jsonl', '_doubao.jsonl')
    with open(output_path, 'w', encoding='utf-8') as output_line:
        with ThreadPoolExecutor(max_workers=32) as executor:
            future_to_line = {
                executor.submit(process_manyidu_v3, line, instruction, ak, model): line for line in tqdm(ll, desc='Submitting Tasks')
            }

            for future in tqdm(as_completed(future_to_line), total=len(future_to_line), desc='Processing Tasks'):
                line = future_to_line[future] 
                try:
                    result = future.result()  
                    line = json.loads(line)
                    line['llm_label'] = result
                    output_line.write(json.dumps(line, ensure_ascii=False) + '\n') 
                except Exception as e:
                    print(f"[ERROR] Exception occurred while processing line {e}")
    print('done')


def split_geci(input_path):
    raw_data = []
    for file in os.listdir(input_path):
        if '.jsonl' in file and 'eng' not in file:
            with open(os.path.join(input_path, file), 'r', encoding='utf-8') as f:
                for line in tqdm(f):
                    try:
                        data = json.loads(line)
                        data = data['gpt_result']
                        if '抱歉' in data:
                            continue
                        else:
                            raw_data.append(data)
                    except:
                        continue
    new_data = []
    for line in raw_data:
        new_data.append(line)
    new_data = list(set(new_data))
    output_list = []
    for line in new_data:
        for l in line.split('|'):
            output_list.append(l)
    output_path = os.path.join(input_path, 'kuochong_data.txt')
    with open(output_path, 'w', encoding='utf-8') as writer:
        for line in output_list:
            writer.write(line+'\n')
    print('done')


if __name__=="__main__":
    manyidu_pipeline(best_model, 'You are an expert in reading comprehension. Carefully analyze each RACE passage and corresponding questions, then choose the most accurate answer from the given options.', 'race_prepared/race_llm_prompts_test.jsonl')
    manyidu_pipeline(best_model, 'You are an expert in reading comprehension. Carefully analyze each RACE passage and corresponding questions, then choose the most accurate answer from the given options.', 'race_prepared/race_llm_prompts_val.jsonl')  