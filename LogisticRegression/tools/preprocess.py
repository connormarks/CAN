from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from .dataset import create_test_train_split
from .config import EMOTION_MAPPING, TOPIC_MAPPING, EKMAN_IDX_TO_EMOTION_MAPPING, MODEL_PATH
import pandas as pd
import numpy as np
import pickle


def create_vectorizer(texts, max_df=0.95, min_df=2, max_features=10000, stop_words='english', save_name=None):
    """
    Vectorizes the text using the TfidfVectorizer

    Inputs:
        texts: list - The text data to vectorize
        max_df: float - The maximum document frequency to remove terms from
        min_df: int - The minimum document frequency to remove terms from
        max_features: int - The maximum number of features to keep
        stop_words: str - The language for stop words

    Returns:
        vectorizer: TfidfVectorizer object - The vectorizer
    """
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=max_features, stop_words=stop_words)
    vectorizer.fit_transform(texts)
    if save_name:
        with open(f'{MODEL_PATH}/{save_name}.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
    return vectorizer


def _map_go_emotion_to_index(row, ignore_neutral=False):
    """
    The go emotion classes, unlike AG News, are text and not indexed.

    This function maps the class to its index for the logistic regression model

    Inputs:
        row: pandas Series - The row of the goemotion data

    Returns:
        class_index: int - The index of the class
    """
    class_index = row.values.tolist().index(1)
    if ignore_neutral and class_index == 27:
        return None
    return class_index


def _apply_ekman_mapping(value):
    """
    Applies the Ekman mapping to the value

    Inputs:
        value: int - The value to apply the Ekman mapping to

    Returns:
        mapped_value: int - The mapped value
    """
    emotion_key = list(EMOTION_MAPPING.keys())[value]
    for ekman_key, replaced_emotions in EKMAN_IDX_TO_EMOTION_MAPPING.items():
        if emotion_key in replaced_emotions:
            return EMOTION_MAPPING[ekman_key]
    return value


def _balance_classes(X, y, sampling_count=1000):
    """
    Balances the classes using the SMOTETomek algorithm

    Inputs:
        X: list - The feature data
        y: list - The target data
        sampling_count: int - The number of samples to take for each class

    Returns:
        X_resampled: list - The resampled feature data
        y_resampled: list - The resampled target data
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


def preprocess_go_data(
        go_data, vectorizer, 
        simplify_with_ekman=False, 
        fix_class_imbalance=False, 
        ignore_neutral=False
    ):
    """
    Preprocesses the goemotion dataset

    Inputs:
        go_data: pandas DataFrame containing the goemotion data
        vectorizer: TfidfVectorizer object
        simplify_with_ekman: boolean indicating whether to simplify the goemotion classes with provided Ekman mapping
        fix_class_imbalance: boolean indicating whether to fix the class imbalance
        ignore_neutral: boolean indicating whether to ignore the neutral class

    Returns:
        X_train: list - The training feature data
        X_test: list - The test feature data
        y_train: list - The training target data
        y_test: list - The test target data
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
    y = y.apply(lambda row: _map_go_emotion_to_index(row, ignore_neutral), axis=1)
    # If any classes are ignored (set to None), remove them from X and y
    mask = y.notna()
    X = X[mask.values]
    y = y[mask].apply(lambda x: int(x))

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

    Inputs:
        ag_data: pandas DataFrame containing the agnews data
        vectorizer: TfidfVectorizer object

    Returns:
        X_train: list - The training feature data
        X_test: list - The test feature data
        y_train: list - The training target data
        y_test: list - The test target data
    """
    descriptions = ag_data["Description"]
    X = vectorizer.transform(descriptions)
    y = ag_data[["Class Index"]]
    y = y.values[:,0].tolist()

    print("\tCreating test train split...")
    X_train, X_test, y_train, y_test = create_test_train_split(X, y)
    print()

    return X_train, X_test, y_train, y_test


def preprocess_custom_dataset(X, y_emotion, y_topic, vectorizer, simplify_with_ekman=False, ignore_neutral=False):
    """
    Preprocesses the custom dataset

    Inputs:
        X: list - The text data to vectorize
        y_emotion: list - The emotion target data
        y_topic: list - The topic target data
        vectorizer: TfidfVectorizer object
        simplify_with_ekman: boolean indicating whether to simplify the classes with provided Ekman mapping

    Returns:
        X: list - The vectorized text data
        y_emotion: list - The emotion target data
        y_topic: list - The topic target data
    """
    y_emotion = [EMOTION_MAPPING[emotion] for emotion in y_emotion]
    y_topic = [TOPIC_MAPPING[topic] for topic in y_topic]

    X = vectorizer.transform(X)
    if simplify_with_ekman:
        print("\tSimplifying classes...")
        y_emotion = map(_apply_ekman_mapping, y_emotion)
        y_emotion = list(y_emotion)

    if ignore_neutral:
        y_emotion = np.array(y_emotion)
        y_topic = np.array(y_topic)

        y_emotion = np.where(y_emotion == 27, None, y_emotion)
        # Mask out the neutral class in X and both targets
        mask = y_emotion != None
        X = X[mask]
        y_topic = y_topic[mask]
        y_emotion = y_emotion[mask]

        y_emotion = y_emotion.tolist()
        y_topic = y_topic.tolist()
    print()

    return X, y_emotion, y_topic
