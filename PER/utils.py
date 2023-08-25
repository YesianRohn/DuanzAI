import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

bert_model = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(bert_model)
VOCAB = ('<PAD>', '[CLS]', '[SEP]', 'O', 'PUNCHLINE')

tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}
MAX_LEN = 256 - 2


class NerDataset(Dataset):
    def __init__(self, f_path):
        self.sents = []
        self.tags_li = []

        with open(f_path, 'r', encoding='utf-8') as f:
            lines = [line.split('\n')[0]
                     for line in f.readlines() if len(line.strip()) != 0]

        tags = [line.split('\t')[1] for line in lines]
        words = [line.split('\t')[0] for line in lines]

        word, tag = [], []
        for char, t in zip(words, tags):
            if char != "&":
                word.append(char)
                tag.append(t)
            else:
                if len(word) > MAX_LEN:
                  self.sents.append(['[CLS]'] + word[:MAX_LEN] + ['[SEP]'])
                  self.tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
                else:
                  self.sents.append(['[CLS]'] + word + ['[SEP]'])
                  self.tags_li.append(['[CLS]'] + tag + ['[SEP]'])
                word, tag = [], []

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx]
        token_ids = tokenizer.convert_tokens_to_ids(words)
        laebl_ids = [tag2idx[tag] for tag in tags]
        seqlen = len(laebl_ids)
        return token_ids, laebl_ids, seqlen

    def __len__(self):
        return len(self.sents)
    
class StrDataset(Dataset):
    def __init__(self, words):
        self.sent = ['[CLS]']
        for char in words:
            self.sent.append(char)
        self.sent.append('[SEP]')
        self.token_tensors = torch.LongTensor(
            [tokenizer.convert_tokens_to_ids(self.sent)])
        self.mask = (self.token_tensors > 0)

def PadBatch(batch):
    maxlen = max([i[2] for i in batch])
    token_tensors = torch.LongTensor(
        [i[0] + [0] * (maxlen - len(i[0])) for i in batch])
    label_tensors = torch.LongTensor(
        [i[1] + [0] * (maxlen - len(i[1])) for i in batch])
    mask = (token_tensors > 0)
    return token_tensors, label_tensors, mask
