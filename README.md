# DuanzAI: Understanding Humor in Slang and Punchlines

## Introduction

DuanzAI aims to examine the humor comprehension ability of LLM towards Chinese slang and enhances it by constructing downstream tasks and prompts.

When we attempt to engage LLM in understanding humor, their cognitive process tends to revolve around identifying punchlines, highlighting the contrasts within these punchlines, and subsequently conducting analysis. In the first and third steps, they demonstrate relatively satisfactory performance on their own; however, errors in the second step can also impact the outcome of the third step. Moreover, if both the first and second steps are accurate, leveraging the extensive knowledge base of large models would generally ensure reliable analysis. 

Consequently, we initiated an experiment to assess the performance of LLMs(ChatGLM-6B and ChatGPT3.5) in identifying punchlines. This was then contrasted with our self-constructed Punchline Entity Recognition (PER) system. Following this, we proceeded with the second step by utilizing a combination of online phonetic matching API and the pinyin2hanzi technique to locate the original words. Subsequently, we integrated PER and the aforementioned components into prompts to observe the enhanced performance of the large model.

## Experiment Result

#### PER-Task

|                      | ChatGLM-6B(zero shot) | GPT3.5 (zero shot) | GPT3.5 (5 shot) | DuanzAI-PER(ours) |
| -------------------- | --------------------- | ------------------ | --------------- | ----------------- |
| Exact Match Accuracy | 0.57                  | 0.87               | 0.92            | 0.97              |
| Exact Match Accuracy | 0.73                  | 0.95               | 0.97            | ——                |

#### Understand-Humor-Task

|           | GPT3.5(zero shot) | GPT3.5 (clue provided) | GPT3.5 (5 shot) |
| --------- | ----------------- | ---------------------- | --------------- |
| Score(HF) | 39.3              | 53.1                   | 51              |



