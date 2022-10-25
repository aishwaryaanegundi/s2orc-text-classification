# Introduction

The approach is to use a pretrained `bert-base-uncased` model and fine tune it to s2orc corpus for the multi-class text classification problem. Each document in the dataset has many relevant fields that could be useful for the model training. However, bert has the limitation of 512 tokens and anything above that is truncated. I chose to use `abstract` and optionally `title` to fine-tune the model.

# Setting up the environment
- Model uses python 3.8.8 which can installed using conda:\
    `conda create --name <env_name> python=3.8.8`
- All other packages required to run the scripts are listed in requirements.txt and can be installed using pip:\
    `pip install -r requirements.txt`
- All the configuration required for accessing, storing data and training the model are stored in `src/config.json`
- train and test data are downloaded and stored in `data/raw/` folder. However, not pushed to repo and is expected to be present at this path. If placed elsewhere, the new path is to be mentioned in config.json
- With this settings, to run the training and evalution, execute the following command from the root folder of the repository:  
    `python ./src/main.py`
- The trained model is saved at `models/bert-base.pt`
- The results of the evaluation is printed on the console

# Results
The computation of macro precision, recall and f1 resulted in the following scores
Metric | Score |
:---:|:---:|
precesion | 0.806 |
recall | 0.768 |
f1 | 0.777 |

# Assumptions and unanswered questions

- `abstract` better represents the paper than other sections of it. The current implementation also includes `title` but the model performance did not improve and in some cases also slightly reduced on including it
- Exploring how `authors` field inclusion and other summary sections like `conclusion`, `future work` affects the performance is unanswered
- Th performance of the model is good but how the hyperparameter tuning on validation set improves it further needs to explored
- In terms of epoch, around epoch 12, the model loss did not improve significantly there after
