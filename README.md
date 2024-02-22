# RUG-D-at-SemEval2024-task8


# Introduction
This Github repository includes the code that we used in our submission to task 8 of SemEval 2024 (https://github.com/mbzuai-nlp/SemEval2024-task8.
Part of our method, is the use of extra generated data. Our generated data can also be found in this repository.

## How to run our scripts

### Subtask A monolingual & Subtask B
These subtasks utilize the same scripts to run/test models. To run the scripts, simply call python3 <script_name>. It is possible to input your own values for hyperparameters and specify different models, by using command-line arguments. For more information, run the -help command-line argument, i.e., python3 model_task_A_mono -help

### Subtask A multilingual
'split.py' is used to split the train and dev set into different languages according to 'source'. 

To run 'fold.py', you can call python3 <script_name> --path <file_name> --model <model_name>. The hyperparameters and the fold number can be changed in the file.

All saved best models for each language are necessary to run 'prediction.py'. The command line 'python3 <script_name>' can easily run.

## Generated Data
