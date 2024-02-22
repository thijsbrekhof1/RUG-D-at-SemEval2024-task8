import json

# input and output files
input_file_path = 'subtaskA_train_multilingual.jsonl'
output_files = {}

# the cource map
source_language_map = {
    'wikipedia': 'en',
    'wikihow': 'en',
    'peerread': 'en',
    'reddit': 'en',
    'arxiv': 'en',
    'arabic': 'ar',
    'russian': 'ru',
    'chinese': 'zh',
    'indonesian': 'id',
    'urdu': 'ur',
    'bulgarian': 'bg',
    'german': 'de',
}

# split the file based on 'source'
with open(input_file_path, 'r', encoding='utf-8') as input_file:
    for line in input_file:
        data = json.loads(line)

        source = data.get('source', 'unknown')

        language_abbr = source_language_map.get(source, 'unknown')

        output_file_path = f'subtaskA_train_{language_abbr}.jsonl'

        if output_file_path not in output_files:
            output_files[output_file_path] = open(output_file_path, 'a', encoding='utf-8')

        output_files[output_file_path].write(json.dumps(data) + '\n')

# close all outout files
for file in output_files.values():
    file.close()
