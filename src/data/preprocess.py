import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from datasets import load_metric
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import json


def __transform_labels(row, label_map):
    row['labels'] = label_map[row['labels'][0]]
    return row

def __combine_fields(row, fields):
    text = ""
    for f in fields:
        text = text + row[f]
    row['text'] = text
    return row

def __preprocess(data, label_map):
    """
    """
    data = data[['title','abstract','mag_field_of_study']]
    data = data.apply(__combine_fields, fields=['title', 'abstract'], axis=1)
    data.rename(columns={'mag_field_of_study': 'labels'}, inplace=True)
    data = data.apply(__transform_labels, label_map = label_map, axis=1)
    data = data[['text', 'labels']]
    return data

tokenizer = None
def __tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def __get_tokenized_dataset(data):
    tokenized_datasets = data.map(__tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")
    return tokenized_datasets.shuffle().select(range(len(data)))


def get_dataloaders(config):
    global tokenizer
    train_data = pd.read_json(config["DATA"]["TRAIN_DATA_PATH"], lines=True)
    test_data = pd.read_json(config["DATA"]["TEST_DATA_PATH"], lines=True)
    
    train_data = __preprocess(train_data,config["DATA"]["LABEL_MAP"])
    test_data = __preprocess(test_data,config["DATA"]["LABEL_MAP"])

    train_dataset = Dataset.from_pandas(train_data)
    test_dataset = Dataset.from_pandas(test_data)

    pretrained_model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(config["TRAINING"]["PRETRAINED_MODEL"], normalization=True)

    train_dataset = __get_tokenized_dataset(train_dataset)
    test_dataset = __get_tokenized_dataset(test_dataset)

    print(train_dataset)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config["TRAINING"]["BATCH_SIZE"])
    test_dataloader = DataLoader(test_dataset, batch_size=config["TRAINING"]["BATCH_SIZE"])

    return train_dataloader, test_dataloader
