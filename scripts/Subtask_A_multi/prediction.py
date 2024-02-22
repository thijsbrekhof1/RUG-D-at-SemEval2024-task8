from datasets import Dataset
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, DataCollatorWithPadding
from langid.langid import LanguageIdentifier, model
from scipy.special import softmax
import os

def preprocess_function(examples, **fn_kwargs):
    return fn_kwargs['tokenizer'](examples["text"], truncation=True, max_length=512)

def test(test_df, model_paths, id2label, label2id, identifier_model):
    predictions_all = pd.DataFrame(columns=['id', 'label'])

    for index, row in test_df.iterrows():
        text = row['text']
        lang = identifier_model.classify(text)[0]

        # If the language is in the specified language list, choose the corresponding model path
        if lang in model_paths:
            model_path = model_paths[lang]
        else:
            model_path = model_paths['others']

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=len(label2id), id2label=id2label, label2id=label2id
        )

        test_dataset = Dataset.from_pandas(pd.DataFrame([row]))
        tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=None,
        )

        # Get model predictions
        predictions = trainer.predict(tokenized_test_dataset)
        prob_pred = softmax(predictions.predictions, axis=-1)
        preds = np.argmax(predictions.predictions, axis=-1)

        # Merge prediction results into a DataFrame
        predictions_df = pd.DataFrame({'id': test_df['id'][index], 'label': preds})
        predictions_all = pd.concat([predictions_all, predictions_df])

    return predictions_all

if __name__ == '__main__':
    # Define a dictionary of model paths, where keys are languages and values are corresponding model paths
    model_paths = {
        'en': "/best_models/en",
        'ar': "/best_models/ar/no_added_text",
        'ru': "/best_models/ru",
        'zh': "/best_models/zh",
        'id': "/best_models/id/no_added_text",
        'ur': "/best_models/ur",
        'bg': "/best_models/bg",
        'de': "/best_models/de/no_added_text",
        'others': "/best_models/multi"
    }
    
    # Define the path to the test file
    test_file_path = "/SemEval2024-Task8/SubtaskA/subtaskA_test_multilingual.jsonl"

    # Read the test data
    test_df = pd.read_json(test_file_path, lines=True)
    
    id2label = {0: "human", 1: "machine"}
    label2id = {"human": 0, "machine": 1}
    
    # Load the language identifier model
    identifier_model = LanguageIdentifier.from_modelstring(model, norm_probs=True)

    # Make predictions
    predictions = test(test_df, model_paths, id2label, label2id, identifier_model)

    # Save prediction results
    output_file_path = "/SemEval2024-Task8/predictions_test.jsonl"
    predictions.to_json(output_file_path, lines=True, orient='records')

    print(f"Predictions saved to: {output_file_path}")