import torch
from torch.utils import data
import os
import warnings
import argparse
import numpy as np
from sklearn import metrics
from models import Bert_BiLSTM_CRF
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from utils import NerDataset, PadBatch, tag2idx, idx2tag
from torch.utils.data import random_split

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train(epoch, model, iterator, optimizer, scheduler, device):
    model.train()
    losses = 0.0
    step = 0
    for batch in iterator:
      step += 1
      tokens, lables, mask = batch
      tokens = tokens.to(device)
      lables = lables.to(device)
      mask = mask.to(device)

      loss = model(tokens, lables, mask)
      losses += loss.item()
      loss.backward()
      optimizer.step()
      scheduler.step()
      optimizer.zero_grad()

    print("Epoch: {}, Loss:{:.4f}".format(epoch, losses/step))


def validate(epoch, model, iterator, device):
    model.eval()
    Y, Y_hat = [], []
    losses = 0
    step = 0
    with torch.no_grad():
        for batch in iterator:
            step += 1
            tokens, lables, mask = batch
            tokens = tokens.to(device)
            lables = lables.to(device)
            mask = mask.to(device)
            y_hat = model(sentence=tokens, mask=mask, is_test=True)
            loss = model(tokens, lables, mask)
            losses += loss.item()
            for j in y_hat:
              Y_hat.extend(j)
            y_mask = (mask == 1)
            y_orig = torch.masked_select(lables, y_mask)
            Y.append(y_orig.cpu())

    Y = torch.cat(Y, dim=0).numpy()
    Y_hat = np.array(Y_hat)
    acc = (Y_hat == Y).mean()*100

    print("Epoch: {}, Val Loss:{:.4f}, Val Acc:{:.3f}%".format(
        epoch, losses/step, acc))
    return model, losses/step, acc


def test(model, iterator, device):
    model.eval()
    Y, Y_hat = [], []
    with torch.no_grad():
        for batch in iterator:
            tokens, lables, mask = batch
            tokens = tokens.to(device)
            lables = lables.to(device)
            mask = mask.to(device)
            y_hat = model(sentence=tokens, mask=mask, is_test=True)
            for j in y_hat:
              Y_hat.extend(j)
            y_mask = (mask == 1)
            y_orig = torch.masked_select(lables, y_mask)
            Y.append(y_orig.cpu())

    Y = torch.cat(Y, dim=0).numpy()
    y_true = [idx2tag[i] for i in Y]
    y_pred = [idx2tag[i] for i in Y_hat]

    return y_true, y_pred


if __name__ == "__main__":

    labels = ['PUNCHLINE']

    best_model = None
    _best_val_loss = 1e18
    _best_val_acc = 1e-18

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="./punchline_data.txt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
      model = Bert_BiLSTM_CRF(tag2idx).cuda()
    else:
      model = Bert_BiLSTM_CRF(tag2idx).cpu()
    torch.manual_seed(args.seed)
    print('Initial model Done...')

    dataset = NerDataset(args.dataset)
    train_size = int(0.7 * len(dataset))
    eval_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - eval_size
    train_dataset, eval_dataset, test_dataset = random_split(
        dataset, [train_size, eval_size, test_size])
    print('Load Data Done...')

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=PadBatch)

    eval_iter = data.DataLoader(dataset=eval_dataset,
                                batch_size=(args.batch_size)//2,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=PadBatch)

    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=(args.batch_size)//2,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=PadBatch)

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-6)

    # Warmup
    len_dataset = len(train_dataset)
    epoch = args.n_epochs
    batch_size = args.batch_size
    total_steps = (len_dataset // batch_size) * \
        epoch if len_dataset % batch_size == 0 else (
            len_dataset // batch_size + 1) * epoch

    warm_up_ratio = 0.1  # Define 10% steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warm_up_ratio * total_steps, num_training_steps=total_steps)

    print(f"\033[32;1mStart Train...\033[0m")
    for epoch in range(1, args.n_epochs+1):

        train(epoch, model, train_iter, optimizer, scheduler, device)
        candidate_model, loss, acc = validate(epoch, model, eval_iter, device)

        if loss < _best_val_loss and acc > _best_val_acc:
          best_model = candidate_model
          _best_val_loss = loss
          _best_val_acc = acc

        print("=============================================")

    y_test, y_pred = test(best_model, test_iter, device)
    torch.save(model, 'duanzai_punchline_per_model.pth')
    print(metrics.classification_report(
        y_test, y_pred, labels=labels, digits=3))
