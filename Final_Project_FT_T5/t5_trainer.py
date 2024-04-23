from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, EarlyStoppingCallback
from transformers import EvalPrediction
from bert_score import score
from datasets import load_metric
from datasets import Dataset, DatasetDict
from transformers import T5ForConditionalGeneration
import pandas as pd
import evaluate
import numpy as np
from tqdm import tqdm
import torch
import re

rouge = evaluate.load('rouge')

torch.manual_seed(12345)
np.random.seed(12345)

# check if gpu is available
device = 'cpu' 
if torch.backends.mps.is_available():
    device = 'mps'
if torch.cuda.is_available():
    device = 'cuda'
print(f"Using '{device}' device")

# model_name = 'google/flan-t5-large'
model_name = 't5-base'

# TODO: Load the tokenizer using AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=512)

# TODO: Load Pre-trained model from HuggingFace Model Hub
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

## Let's see how many parameters we are going to be changing
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

print_trainable_parameters(model)

train_df = pd.read_csv('data/train_set.csv', sep=',')
test_df = pd.read_csv('data/test_set.csv', sep=',')
dev_df = pd.read_csv('data/dev_set.csv', sep=',')

train_df['content'] = train_df['content'].apply(lambda x: re.sub(r'[\r\n]+', ' ', x))
test_df['content'] = test_df['content'].apply(lambda x: re.sub(r'[\r\n]+', ' ', x))
dev_df['content'] = dev_df['content'].apply(lambda x: re.sub(r'[\r\n]+', ' ', x))

labels = ['title']
target_col = ['data_id', 'content', 'title']    

train_ds = Dataset.from_pandas(train_df[target_col])    
dev_ds = Dataset.from_pandas(dev_df[target_col])
test_ds = Dataset.from_pandas(test_df[target_col].iloc[:1200])  

dataset_dict = DatasetDict({    
    'train': train_ds,
    'dev': dev_ds,
    'test': test_ds
})

def preprocess_function(examples):
    # Prepends the string "summarize: " to each document in the 'text' field of the input examples.
    # This is done to instruct the T5 model on the task it needs to perform, which in this case is summarization.
    inputs = ["Generate title for following content: " + doc for doc in examples["content"]]

    # Tokenizes the prepended input texts to convert them into a format that can be fed into the T5 model.
    # Sets a maximum token length of 1024, and truncates any text longer than this limit.
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, return_tensors="pt", padding='longest')

    # Tokenizes the 'summary' field of the input examples to prepare the target labels for the summarization task.
    # Sets a maximum token length of 128, and truncates any text longer than this limit.
    labels = tokenizer(text_target=examples["title"], max_length=32, truncation=True, return_tensors="pt", padding='longest')

    # Assigns the tokenized labels to the 'labels' field of model_inputs.
    # The 'labels' field is used during training to calculate the loss and guide model learning.
    model_inputs["labels"] = labels["input_ids"]

    # Returns the prepared inputs and labels as a single dictionary, ready for training.
    return model_inputs

tokenized_datasets = dataset_dict.map(preprocess_function, batched=True)

training_args = Seq2SeqTrainingArguments(
    seed = 12345, 
    do_eval=True,
    output_dir="my_fine_tuned_t5_model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    # weight_decay=0.01,
    num_train_epochs=3,
    eval_steps=210,
    save_steps=420,
    evaluation_strategy="steps",       
    save_strategy="steps",  
    load_best_model_at_end=True, 
    predict_with_generate=True,
    metric_for_best_model="rouge2",
    greater_is_better=True,
    # fp16=True,
    report_to='wandb',
    logging_dir='./t5_logs',
    run_name="t5-base-title-summarizer",  
    fp16=True
)

def compute_metrics(eval_pred: EvalPrediction):
    # Unpacks the evaluation predictions tuple into predictions and labels.
    predictions, labels = eval_pred

    # Decodes the tokenized predictions back to text, skipping any special tokens (e.g., padding tokens).
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replaces any -100 values in labels with the tokenizer's pad_token_id.
    # This is done because -100 is often used to ignore certain tokens when calculating the loss during training.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decodes the tokenized labels back to text, skipping any special tokens (e.g., padding tokens).
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Computes the ROUGE metric between the decoded predictions and decoded labels.
    # The use_stemmer parameter enables stemming, which reduces words to their root form before comparison.
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # Calculates the length of each prediction by counting the non-padding tokens.
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]

    # Computes the mean length of the predictions and adds it to the result dictionary under the key "gen_len".
    result["gen_len"] = np.mean(prediction_lens)

    # Rounds each value in the result dictionary to 4 decimal places for cleaner output, and returns the result.
    return {k: round(v, 4) for k, v in result.items()}


t5_trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
)

t5_trainer.train()
t5_trainer.evaluate()


test_title = t5_trainer.predict(tokenized_datasets['test']) 
print(test_title)
result = []
for title_ids in test_title.predictions:
    result.append(tokenizer.decode(title_ids, skip_special_tokens=True))
print(result[0])
test_set = test_df[['data_id', 'title']].iloc[:1200].copy()
test_set['predicted_title'] = result
test_set.to_csv('data/test_set_summary_t5_large.csv', index=False) 

# Load the Score
print("Rouge Score: ")
print(rouge.compute(predictions=test_set['predicted_title'], references=test_set['title'], use_stemmer=True))
print("Bert Score: ")
P, R, F1 = score(test_set['predicted_title'].to_list(), test_set['title'].to_list(), lang='en', verbose=True)
print(f'Precision: {P.mean():.4f}')
print(f'Recall: {R.mean():.4f}')
print(f'F1: {F1.mean():.4f}')