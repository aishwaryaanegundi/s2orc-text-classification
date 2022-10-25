from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler
import torch
from datasets import load_metric


def run(test_dataloader, config, metrics, average = None):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    path_to_saved_model = config["DATA"]["SAVED_MODELS_PATH"]+"bert-base.pt"
    pretrained_model_name = config["TRAINING"]["PRETRAINED_MODEL"]
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=5)
    model.load_state_dict(torch.load(path_to_saved_model))
    model.to(device)
    metric= load_metric(metrics)
    model.eval()
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    score = metric.compute(average='macro')
    return score