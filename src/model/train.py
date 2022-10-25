import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

def run(train_dataloader, config):
    num_epochs = config["TRAINING"]["NUM_EPOCHS"]
    batch_size = config["TRAINING"]["BATCH_SIZE"]
    learning_rate = config["TRAINING"]["LEARNING_RATE"]
    path_to_save_model = config["DATA"]["SAVED_MODELS_PATH"]+"bert-base.pt"
    pretrained_model_name = config["TRAINING"]["PRETRAINED_MODEL"]

    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=5)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    loss = 0
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            loss = loss
        print(epoch, ":", loss)
    torch.save(model.state_dict(), path_to_save_model)
