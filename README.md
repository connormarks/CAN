# CAN
CS 175 Project in AI 

## 1. Install Requirements
Run the following command in order to install all dependencies required to run our model:\
```pip install -r requirements.txt```

Note: the `FewShotLLM`, `Logistic Regression`, and `SyntheticDataGeneration` tools each require their own requirements and Python versions (and should be run with their own venv).


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
   * [ollama](https://pypi.org/project/ollama/)
   * [imbalanced-learn](https://pypi.org/project/imbalanced-learn/)
   * [sentence-transformers](https://pypi.org/project/sentence-transformers/)
   * [google-genai](https://pypi.org/project/google-genai/)

We also created a custom library for shared tools. It can be found in `/Src/Shared/custom_llm_tools`
    

## Publicly available codes used:
Referenced [this cookbook notebook](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Streaming.ipynb) from Google to get started wiht prompting Gemini models. This code was used as a starting point in `/Src/Shared/custom_llm_tools/gemini_api.py`


## Scripts/functions written by our team:
   * Src/BERT/config.py Stores globals used in the training loop (25 lines)
   * Src/BERT/dataset.py Preprocesses the two datasets, including tokenization and adding positonal weights (148 lines)
   * Src/BERT/evaluate.py Calculates our evaluation metrics during training such as accuracy and F1 scores (166 lines)
   * Src/BERT/model.py Initializes the BERT and two neural net heads, contains the forward pass (38 lines)
   * Src/BERT/train.py The main BERT training loop, loss calculation and a function for validation metrics (180 lines)
   * Src/run/run.py Initilizes the training loop and manages small output variables into files during training (45 lines)
   * Src/test/test.py Deploys the model to run with Run 9's pytorch model weights, on the merged test dataset (166 lines)
   * Everything in Src/Shared, shared LLM and custom dataset eval tooling which was condensed into a custom library (481 lines total)
     * Includes code for visualizing LLM dataset metrics, using Gemini api, using Ollama api, and handling model context
   * Everything in Src/Baselines/FewShotLLM, includes code to run and evaluate a few-shot baselines as well as prompts for the model (400 lines total)
     * response_format.py A pythonic way of enforcing the output formatting for the Ollama api (25 lines)
     * run_few_shot.py Helpers to setup a few-shot run, as well as the code to get user input and evaluate the run (214 lines)
     * scoring.py Scoring functions, similar to those in the Logistic Regression baseline, to score the model's performance (99 lines)
     * config.py Has mappings for classes to their indicies, input and output folders, and other config stuff (62 lines)
   * Everything in Src/Baselines/LogisticRegression, includes code to train, run, and evaluate two logistic regression models, one for emotion and one for topic. (724 lines total)
     * /tools Includes config (config.py), dataset loading and preprocessing (dataset.py and preprocess.py), training (train.py), and scoring (scoring.py) (549 lines total)
     * advanced_run.py Automatically trains, runs, and evaluates the two models on their performance across the public datasets and our custom dataset (105 lines)
     * subjective_test.py Allows the user to input text for classification, allowing us to evaluate our models subjectively (38 lines)
   * Everything in Src/SyntheticDataGeneration, includes code to generate synthetic data using Ollama or Gemini, manually validate the output, and visualize the class balance (450 lines total)
     * auto_generate.py The primary file used to generate and automatically verify all examples, only used with higher quality models to generate thousands of examples (157 lines)
     * show_stats.py Used to show the class balance of the resulting dataset after verifying (if needed) (72 lines)
