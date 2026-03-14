# CAN
CS 175 Project in AI 

## 1. Install Requirements
Run the following command in order to install all dependencies required to run our model:\
```pip install -r requirements.txt```



## Libraries used:
   * [transformers](https://huggingface.co/docs/transformers/quicktour)
   * [BERT](https://huggingface.co/google-bert/bert-base-uncased)
   * [pytorch](https://docs.pytorch.org/docs/stable/pytorch-api.html)
   * [pandas](https://pandas.pydata.org/docs/reference/index.html)
   * [kagglehub](https://github.com/Kaggle/kagglehub)
   * [numpy](https://numpy.org/devdocs/reference/index.html)
   * [sklearn](https://scikit-learn.org/stable/api/index.html)
   * [matplotlib](https://matplotlib.org/stable/api/index.html)
   * [seaborn](https://seaborn.pydata.org/api.html)
    

## Publicly available codes used:


## Scripts/functions written by our team:
   * Src/BERT/config.py Stores globals used in the training loop (25 lines)
   * Src/BERT/dataset.py Preprocesses the two datasets, including tokenization and adding positonal weights (148 lines)
   * Src/BERT/evaluate.py Calculates our evaluation metrics during training such as accuracy and F1 scores (166 lines)
   * Src/BERT/model.py Initializes the BERT and two neural net heads, contains the forward pass (38 lines)
   * Src/BERT/train.py The main BERT training loop, loss calculation and a function for validation metrics (180 lines)
   * Src/run/run.py Initilizes the training loop and manages small output variables into files during training (45 lines)
   * Src/test/test.py Deploys the model to run with Run 9's pytorch model weights, on the merged test dataset (166 lines)
