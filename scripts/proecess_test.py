import json
raw_data = []
with open('LLM_out/gpt4o_1124/race_llm_prompts_val_doubao.jsonl', 'r', encoding='utf-8') as reader:
    for line in reader:
        raw_data.append(json.loads(line))
with open('LLM_out/gpt4o_1124/race_llm_prompts_val_doubao_p.jsonl', 'w', encoding='utf-8') as writer:
    for line in raw_data:
        line['pred_answer'] = line['llm_label'].split('.')[0]
        writer.write(json.dumps(line, ensure_ascii=False) + '\n')