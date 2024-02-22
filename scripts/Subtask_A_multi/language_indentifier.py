import json

# 输入和输出文件
input_file_path = 'subtaskA_train_multilingual.jsonl'
output_files = {}

# 映射source到语言缩写
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

with open(input_file_path, 'r', encoding='utf-8') as input_file:
    for line in input_file:
        # 解析JSON数据
        data = json.loads(line)

        # 获取文本和源信息
        source = data.get('source', 'unknown')

        # 获取语言缩写
        language_abbr = source_language_map.get(source, 'unknown')

        # 构建输出文件路径
        output_file_path = f'subtaskA_train_{language_abbr}.jsonl'

        # 如果输出文件尚未打开，则打开它并将其保存在字典中
        if output_file_path not in output_files:
            output_files[output_file_path] = open(output_file_path, 'a', encoding='utf-8')

        # 将记录写入相应的文件
        output_files[output_file_path].write(json.dumps(data) + '\n')

# 关闭所有输出文件
for file in output_files.values():
    file.close()
