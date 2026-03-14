# Dataset tools for the logistic regression baseline
# Authors: Nathan Pietrantonio
from sklearn.model_selection import train_test_split
from .config import EMOTION_MAPPING, TOPIC_MAPPING
import pandas as pd
import kagglehub
import json


def load_datasets():
    """
    Loads the datasets from the repo and returns their contents

    Returns:
        go_data: pandas DataFrame containing the goemotion data
        ag_data: pandas DataFrame containing the agnews data
    """
    print("Loading goemotion dataset...")
    goemotion_path = kagglehub.dataset_download("debarshichanda/goemotions")
    print(f"Goemotion dataset loaded from {goemotion_path}\n")
    print("Loading agnews dataset...")
    agnews_path = kagglehub.dataset_download("amananandrai/ag-news-classification-dataset")
    print(f"Agnews dataset loaded from {agnews_path}\n")
    return pd.read_csv(f"{goemotion_path}/data/full_dataset/goemotions_1.csv"), pd.read_csv(f"{agnews_path}/train.csv")


def create_test_train_split(X, y, test_size=0.2, random_state=42):
    """
    Creates a test train split

    Inputs:
        X: list - The feature data
        y: list - The target data
        test_size: float - The size of the test set
        random_state: int - The random state to use for the split

    Returns:
        X_train: list - The training feature data
        X_test: list - The test feature data
        y_train: list - The training target data
        y_test: list - The test target data
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
