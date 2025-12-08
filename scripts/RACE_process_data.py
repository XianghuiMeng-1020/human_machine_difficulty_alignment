import os
from tqdm import tqdm

def process_data2jsonl(input_folder, output_path):
    with open(output_path, 'w', encoding='utf-8') as writer:
        for file in tqdm(os.listdir(input_folder)):
            with open(os.path.join(input_folder, file), 'r', encoding='utf-8') as reader:
                for line in reader:
                    writer.write(line+'\n')

if __name__ == "__main__":
    process_data2jsonl('data/RACE/train/high', 'data/RACE/train_high.jsonl')
    process_data2jsonl('data/RACE/dev/high', 'data/RACE/dev_high.jsonl')
    process_data2jsonl('data/RACE/dev/middle', 'data/RACE/dev_mid.jsonl')
    process_data2jsonl('data/RACE/test/high', 'data/RACE/test_high.jsonl')
    process_data2jsonl('data/RACE/test/middle', 'data/RACE/test_mid.jsonl')