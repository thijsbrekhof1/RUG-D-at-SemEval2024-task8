# -*- coding: utf-8 -*-

from datasets import Dataset
import pandas as pd
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, \
    AutoTokenizer, set_seed
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from scipy.special import softmax
import argparse
import logging
from functools import partial
import shutil
import pandas as pd
import logging.handlers
import sys

sys.path.append('.')


def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs['tokenizer'](examples["text"], truncation=True, max_length=512)


def get_data(data_path, random_seed):
    df = pd.read_json(data_path, lines=True)

    # Use StratifiedKFold for stratified splitting
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_seed)

    for fold, (train_index, val_index) in enumerate(skf.split(df, df['label'])):
        train_fold, val_fold = df.iloc[train_index], df.iloc[val_index]

        # Save the training and validation datasets to the specified folder
        fold_folder = f"save/{model_name}/subtask{subtask}/{random_seed}/fold{fold + 1}/best"
        os.makedirs(fold_folder, exist_ok=True)  # Create the folder

        train_fold.to_json(os.path.join(fold_folder, 'train.jsonl'), orient='records', lines=True)
        val_fold.to_json(os.path.join(fold_folder, 'dev.jsonl'), orient='records', lines=True)

        yield train_fold, val_fold


def compute_metrics(eval_pred):
    f1_metric = evaluate.load("f1")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    results = {}
    results.update(f1_metric.compute(predictions=predictions, references=labels, average="micro"))

    return results


def fine_tune(train_df, valid_df, checkpoints_path, id2label, label2id, model):
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSequenceClassification.from_pretrained(
        model, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True,
                                                fn_kwargs={'tokenizer': tokenizer, 'max_length': 512})
    tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True,
                                                fn_kwargs={'tokenizer': tokenizer, 'max_length': 512})

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    best_model_path = checkpoints_path + '/best/'

    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)

    trainer.save_model(best_model_path)


def test(data_df, model_name, id2label, label2id, fold):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    data_dataset = Dataset.from_pandas(data_df)

    tokenized_data_dataset = data_dataset.map(partial(preprocess_function, tokenizer=tokenizer), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    predictions = trainer.predict(tokenized_data_dataset)
    prob_pred = softmax(predictions.predictions, axis=-1)
    preds = np.argmax(predictions.predictions, axis=-1)
    metric = evaluate.load("bstrai/classification_report")
    results = metric.compute(predictions=preds, references=predictions.label_ids)

    # Save predictions for the fold
    predictions_df = pd.DataFrame({'id': data_df['id'], 'label': preds})
    predictions_path = f"{model_name}/predictions.jsonl"
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)  # Ensure the directory exists
    predictions_df.to_json(predictions_path, lines=True, orient='records')

    return results


def evaluates(pred_fpath, gold_fpath):
    """
      Evaluates the predicted classes w.r.t. a gold file.
      Metrics are: f1-macro, f1-micro and accuracy

      :param pred_fpath: a json file with predictions,
      :param gold_fpath: the original annotated gold file.

      The submission of the result file should be in jsonl format.
      It should be a lines of objects:
      {
        id     -> identifier of the test sample,
        labels -> labels (0 or 1 for subtask A and from 0 to 5 for subtask B),
      }
    """

    pred_labels = pd.read_json(pred_fpath, lines=True)[['id', 'label']]
    gold_labels = pd.read_json(gold_fpath, lines=True)[['id', 'label']]

    merged_df = pred_labels.merge(gold_labels, on='id', suffixes=('_pred', '_gold'))

    macro_f1 = f1_score(merged_df['label_gold'], merged_df['label_pred'], average="macro", zero_division=1)
    micro_f1 = f1_score(merged_df['label_gold'], merged_df['label_pred'], average="micro", zero_division=1)
    accuracy = accuracy_score(merged_df['label_gold'], merged_df['label_pred'])

    return macro_f1, micro_f1, accuracy


def evaluate_fold_results(model_name, subtask, random_seed, folds):
    macro_f1_scores = []
    micro_f1_scores = []
    accuracy_scores = []

    for fold in range(1, folds + 1):
        try:
            # Adjust paths based on the actual structure
            predictions_path = f"save/{model_name}/subtask{subtask}/{random_seed}/fold{fold}/best/predictions.jsonl"
            gold_path = f"save/{model_name}/subtask{subtask}/{random_seed}/fold{fold}/best/dev.jsonl"

            results = evaluates(predictions_path, gold_path)
            print(f'fold:{fold}')
            print(f'accuracy:{results[2]}')
            macro_f1_scores.append(results[0])
            micro_f1_scores.append(results[1])
            accuracy_scores.append(results[2])
        except Exception as e:
            print(f"Error occurred in Fold {fold}: {type(e).__name__} - {e}")

    # Add condition check to avoid division by zero
    if len(macro_f1_scores) > 0:
        avg_macro_f1 = sum(macro_f1_scores) / len(macro_f1_scores)
        std_macro_f1 = np.std(macro_f1_scores)
    else:
        avg_macro_f1 = 0.0  # or any default value you consider appropriate
        std_macro_f1 = 0.0

    if len(micro_f1_scores) > 0:
        avg_micro_f1 = sum(micro_f1_scores) / len(micro_f1_scores)
        std_micro_f1 = np.std(micro_f1_scores)
    else:
        avg_micro_f1 = 0.0  # or any default value you consider appropriate
        std_micro_f1 = 0.0

    if len(accuracy_scores) > 0:
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        std_accuracy = np.std(accuracy_scores)
    else:
        avg_accuracy = 0.0  # or any default value you consider appropriate
        std_accuracy = 0.0

    print(f"\nAverage Scores Across {folds} Folds:")
    print(f"Average Macro F1: {avg_macro_f1} +- {std_macro_f1}")
    print(f"Average Micro F1: {avg_micro_f1} +- {std_micro_f1}")
    print(f"Average Accuracy: {avg_accuracy} +- {std_accuracy}")



parser = argparse.ArgumentParser()
parser.add_argument("--path", "-path", required=True, help="Path to the language file.", type=str)
parser.add_argument("--model", "-m", required=True, help="Transformer to train and test", type=str)

args = parser.parse_args()

path = args.path
model_name = args.model

drive_path = '/SemEval2024-Task8/'
data_path = drive_path + path
subtask = 'A'
random_seed = 0
folds = 10

if os.path.exists("save"):
    shutil.rmtree("save")

if not os.path.exists(data_path):
    logging.error("File doesn't exist: {}".format(data_path))
    raise ValueError("File doesn't exist: {}".format(data_path))

if subtask == 'A':
    id2label = {0: "human", 1: "machine"}
    label2id = {"human": 0, "machine": 1}
elif subtask == 'B':
    id2label = {0: 'human', 1: 'chatGPT', 2: 'cohere', 3: 'davinci', 4: 'bloomz', 5: 'dolly'}
    label2id = {'human': 0, 'chatGPT': 1, 'cohere': 2, 'davinci': 3, 'bloomz': 4, 'dolly': 5}
else:
    logging.error("Wrong subtask: {}. It should be A or B".format(subtask))
    raise ValueError("Wrong subtask: {}. It should be A or B".format(subtask))

set_seed(random_seed)

for fold, (train_df, valid_df) in enumerate(get_data(data_path, random_seed)):

    if fold >= 1:
        fold_to_delete = fold
        base_path = f"save/{model_name}/subtask{subtask}/{random_seed}/fold{fold_to_delete}/"

        # Find and delete folders starting with 'checkpoint'
        for entry in os.listdir(base_path):
            path_to_delete = os.path.join(base_path, entry)
            if os.path.isdir(path_to_delete) and entry.startswith('checkpoint'):
                shutil.rmtree(path_to_delete)
                print(f"Deleted unnecessary checkpoint files at {path_to_delete}")

    print(f"\nTraining Fold {fold + 1}")
    fine_tune(train_df, valid_df, f"save/{model_name}/subtask{subtask}/{random_seed}/fold{fold + 1}", id2label,
              label2id, model_name)
    
    print(f"\nTesting Fold {fold + 1}")
    results = test(valid_df, f"save/{model_name}/subtask{subtask}/{random_seed}/fold{fold + 1}/best/", id2label,
                   label2id, fold + 1)

    logging.info(f"\nResults for Fold {fold + 1}:\n{results}")

# Evaluate results across all folds
evaluate_fold_results(model_name, subtask, random_seed, folds)
