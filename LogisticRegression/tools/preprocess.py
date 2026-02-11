from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from .dataset import create_test_train_split
from .config import EMOTION_MAPPING, EKMAN_IDX_TO_EMOTION_MAPPING
import pandas as pd
import numpy as np


def create_vectorizer(texts, max_df=0.95, min_df=2, max_features=10000, stop_words='english'):
    """
    Vectorizes the text using the TfidfVectorizer
    """
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=max_features, stop_words=stop_words)
    vectorizer.fit_transform(texts)
    return vectorizer


def _map_go_emotion_to_index(row):
    """
    The go emotion classes, unlike AG News, are text and not indexed.

    This function maps the class to its index for the logistic regression model
    """
    class_index = row.values.tolist().index(1)
    return class_index


def _apply_ekman_mapping(value):
    """
    Applies the Ekman mapping to the value
    """
    emotion_key = list(EMOTION_MAPPING.keys())[value]
    for ekman_key, replaced_emotions in EKMAN_IDX_TO_EMOTION_MAPPING.items():
        if emotion_key in replaced_emotions:
            return EMOTION_MAPPING[ekman_key]
    return value


def _balance_classes(X, y, sampling_count=1000):
    """
    Balances the classes using the SMOTETomek algorithm
    """
    print("\tBalancing classes...")
    # smote = SMOTE(random_state=42)
    # X_resampled, y_resampled = smote.fit_resample(X, y)

    # Create a dict for the number of samples to take for each class
    classes, counts = np.unique(y, return_counts=True)
    sampling_strategy = {}
    for cls, count in zip(classes, counts):
        sampling_strategy[cls] = min(count, sampling_count)

    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    X_temp, y_temp = rus.fit_resample(X, y)

    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_temp, y_temp)

    return X_resampled, y_resampled


def preprocess_go_data(go_data, vectorizer, simplify_with_ekman=False, fix_class_imbalance=False):
    """
    Preprocesses the goemotion dataset
    """
    # Filter out unclear examples
    go_data = go_data[go_data["example_very_unclear"] == False]

    columns_to_ignore = [ 
        "id", 
        "author", 
        "subreddit", 
        "link_id", 
        "parent_id", 
        "created_utc", 
        "rater_id", 
        "example_very_unclear"
    ]

    # Create X and y
    texts = go_data["text"]
    X = vectorizer.transform(texts)
    y = go_data.drop(columns=["text"] + columns_to_ignore)
    # Map the emotion to its index for the linear regression model
    y = y.apply(lambda row: _map_go_emotion_to_index(row), axis=1)

    if simplify_with_ekman:
        print("\tSimplifying classes...")
        y = y.apply(_apply_ekman_mapping)
    
    y = y.values.tolist()

    print("\tCreating test train split...")
    X_train, X_test, y_train, y_test = create_test_train_split(X, y)

    if fix_class_imbalance:
        X_train, y_train = _balance_classes(X_train, y_train)
    print()

    return X_train, X_test, y_train, y_test


def preprocess_ag_data(ag_data, vectorizer):
    """
    Preprocesses the agnews dataset
    """
    descriptions = ag_data["Description"]
    X = vectorizer.transform(descriptions)
    y = ag_data[["Class Index"]]
    y = y.values[:,0].tolist()

    print("\tCreating test train split...")
    X_train, X_test, y_train, y_test = create_test_train_split(X, y)
    print()

    return X_train, X_test, y_train, y_test


def preprocess_custom_dataset(X, y, vectorizer, simplify_with_ekman=False):
    """
    Preprocesses the custom dataset
    """
    X = vectorizer.transform(X)
    print("\tSimplifying classes...")
    if simplify_with_ekman:
        y = map(_apply_ekman_mapping, y)
        y = list(y)
    print()
    return X, y
