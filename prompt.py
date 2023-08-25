import openai
import json
import time
import argparse

parser = argparse.ArgumentParser(
    description='Select a prompt type for the task')
parser.add_argument(
    '--task', help='The Task for ChatGPT to Prompt', choices=['1', '2'], default='1')
parser.add_argument(
    '--batch', help='The Data Batch for Processing Once, Value: 1-100', type=int, default=1)
parser.add_argument(
    '--type', help='The Prompt Type for Each Task: [Task1: 1 means zero-shot, 2 means 5-shot], [Task2: 1 means clue-provided, 2 means 5-shot]', choices=['1', '2'], default='1')
args = parser.parse_args()

## add your openai key
openai.api_key = ""

with open('data/task_1.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

count = 0
messages = []
message = ""
template = ""
if args.task == '1':
    if args.type == '1':
        template = "你现在的任务是从每行笑话中找出其中幽默来源的那一个词语，需要注意到是：当你无法找到时请输出未知；保证每行输出只有一个词语；每行输出加上对应笑话的标号。"
    else:
        template = "你现在的任务是从每行笑话中找出其中幽默来源的那一个词语，需要注意到是：当你无法找到该行笑话请输出未知；保证每行输出只有一个词语；每行输出加上对应笑话的标号。示例1：{输入：老王一看就知道柿子是坏的，因为老王有一双透柿眼。 输出：透视眼}；示例2：{输入：制取氧气的时候，小王踏不下心来，老师说他：“心浮气造！” 输出：心浮气造}；示例3：{输入：“能卖给我一些制作锁头的原料吗？”“不好意思，我们不出锁料。”  输出：不出锁料}；示例4：{输入：实验室的老师让学生去储藏室领一些细菌，这个学生成了领菌人物。  输出：领菌人物}；示例5：{输入：谁中了这颗子弹可就倒霉了，因为这颗子弹是倒霉弹。 输出：倒霉弹}"

for item in data:
    count += 1
    item = data[count]
    if args.task == '1':
        message = message + str(count) + item['text']
    elif args.type == '1':
        message = f"“{item['text']}”这是一句笑话，其中的笑点在“{item['punchline']}”，请你先找出笑点的谐音词，在通过谐音的反差，分析幽默所在"
    else:
        message = "这里有五个示例。示例1：{输入：老王没有温度感觉，身后都着火了，全燃不知。 分析：这是一句谐音梗，“全燃不知”是“全然不知”的谐音。在这句笑话的描述中，老王燃起来了却没有感觉，解释了为什么全燃不知，与谐音词全然不知意义不同，带来了幽默的效果。}；示例2：{输入：气象局把大象给气死了。 分析：这是一句双关梗，气象局本来是发布天气的机构，但把气和象拆开来可以理解为使大象生气，由此产生了歧义，产生了幽默的效果。}；示例3：{输入：“都说江南水香，我怎么闻不到？”  分析：这是一句谐音梗，“江南水香”是“江南水乡”的谐音，“水乡”本身是对江南流水美景的赞美，“乡”被谐音为“香”后意思变成了水有香味，由此产生了幽默的效果。}；示例4：{输入：啊！矩阵！啊！行列式！啊！特征向量！这是一首《线代诗》。  分析：这是一句谐音梗，前面是一首和线性代数（简称为线代）有关的诗，因此可被称为”线代诗”，其刚好为“现代诗”的谐音，带来了幽默的效果。}；示例5：{输入：宅是一种房沉迷。  分析：这是一句谐音梗，宅指的是一直在沉迷在房间中不出去，因此可以叫“房沉迷”，其又是“防沉迷”的谐音，产生了幽默的效果。}。以上的示例是对每句笑话的幽默分析，接下来我会给一句笑话，你的任务是理解和分析幽默所在。" \
            + item['text']
    if count % args.batch == 0 or count >= len(data):
        message += template
        messages.append({"role": "user", "content": message})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        reply = response["choices"][0]["message"]["content"]
        print(reply)
        time.sleep(20)
        messages = []
        message = ""