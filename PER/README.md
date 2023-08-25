# Punchline Entity Recognition

This directory contains code for a Punchline Entity Recognition (PER) task. PER aims to identify the punchline, including homophones, phonetic characteristics, and other linguistic features in Chinese jokes. The task is inspired for the suboptimal performance of LLM in handling such elements. 

## Getting Started

Follow these steps to set up and run the NER task code:

1. Install the required dependencies by running the following command:

   ```
   pip install -r requirements.txt
   ```

2. Generate training data:

   Run the `nerdata.py` script in the same directory. This script takes labeled training data and converts it into a format suitable for training the PER model.

3. Start Training:

   Once having the preprocessed training data, you can start training the PER model. Run the `main.py` script, which collaborates with `models.py` and `utils.py` to train the model.

4. Hyperparameter Tuning:

   Hyperparameter tuning has been performed using `wandb_main.py`. The best model's hyperparameters and training bash are: 

   ```
   python main.py --n_epochs 8 --lr 0.002158 --batch_size 4
   ```

   

## Model Performance

After training the model(duanzai_punchline_per_model.pth), the performance on the test set and punchline recall rate are as follows:

|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| PUNCHLINE    | 0.970     | 0.969  | 0.969    | 1375    |
| micro avg    | 0.970     | 0.969  | 0.969    | 1375    |
| macro avg    | 0.970     | 0.969  | 0.969    | 1375    |
| weighted avg | 0.970     | 0.969  | 0.969    | 1375    |

|           | Recall | Performance |
| --------- | ------ | ----------- |
| Punchline | 97.2%  | (2478/2549) |

## Generating Jokes

The `test.py` script to demonstrate the model's capabilities. This script reads joke data and generates the punchline for each joke. If a punchline cannot be generated, an empty line is output.

