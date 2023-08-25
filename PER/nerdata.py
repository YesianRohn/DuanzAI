import json

# 从JSON文件中加载数据
with open('../data/task_1.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 初始化NER数据列表
ner_data = []

# 遍历每个数据项
for item in data:
    text = item['text']
    punchline = item['punchline']
    # 创建一个NER样本，将文本拆分为单个字符，标记PUNCHLINE实体
    ner_sample = []
    index = text.find(punchline)
    if index != -1:
        for i in range(index):
            ner_sample.append((text[i], 'O'))
        for char in punchline:
            ner_sample.append((char, 'PUNCHLINE'))
        for i in range(index + len(punchline), len(text)):
            ner_sample.append((text[i], 'O'))
    else:
        for char in text:
            ner_sample.append((char, 'O'))  # 'O' 表示非实体部分

    ner_data.append(ner_sample)

# 将NER数据保存到文件中
with open('punchline_data.txt', 'w', encoding='utf-8') as f:
    for ner_sample in ner_data:
        for char, label in ner_sample:
            f.write(f"{char}\t{label}\n")
        f.write('&	O\n')
        f.write('\n')  # 每个样本之间空一行