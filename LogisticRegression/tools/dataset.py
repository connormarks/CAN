from sklearn.model_selection import train_test_split
from .config import EMOTION_MAPPING, TOPIC_MAPPING
import pandas as pd
import kagglehub
import json


def load_datasets():
    """
    Loads the datasets from the repo and returns their contents
    """
    print("Loading goemotion dataset...")
    goemotion_path = kagglehub.dataset_download("debarshichanda/goemotions")
    print(f"Goemotion dataset loaded from {goemotion_path}\n")
    print("Loading agnews dataset...")
    agnews_path = kagglehub.dataset_download("amananandrai/ag-news-classification-dataset")
    print(f"Agnews dataset loaded from {agnews_path}\n")
    return pd.read_csv(f"{goemotion_path}/data/full_dataset/goemotions_1.csv"), pd.read_csv(f"{agnews_path}/train.csv")


def load_custom_dataset(path):
    """
    Loads our custom joint dataset for eval
    """
    with open(path, 'r') as file:
        data = json.load(file)

    X = [obj['text'] for obj in data]
    y_emotion = [obj['emotion'][0] for obj in data]
    y_topic = [obj['topic'] for obj in data]

    y_emotion = [EMOTION_MAPPING[emotion] for emotion in y_emotion]
    y_topic = [TOPIC_MAPPING[topic] for topic in y_topic]

    return X, y_emotion, y_topic


def create_test_train_split(X, y, test_size=0.2, random_state=42):
    """
    Creates a test train split of the data
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
