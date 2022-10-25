from data import preprocess
from model import train
from model import evaluate
import json

def __read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def main():
    # Get Configurations
    config = __read_json("./src/config.json")

    # Get Data loaders
    train_dataloader, test_dataloader = preprocess.get_dataloaders(config)

    #Train the Model
    train.run(train_dataloader, config)

    # #Evaluate the model
    precision = evaluate.run(test_dataloader, config, "precision")
    recall = evaluate.run(test_dataloader, config, "recall")
    f1 = evaluate.run(test_dataloader, config, "f1")
    print(precision, recall, f1)

if __name__ == "__main__":
    main()