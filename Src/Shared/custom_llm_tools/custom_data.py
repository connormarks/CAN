# Authors: Nathan Pietrantonio
import json
import numpy as np


def apply_ekman_mapping(value, emotion_mapping, ekman_idx_to_emotion_mapping):
    """
    Applies the Ekman mapping to the value

    Inputs:
        value: int - The value to apply the Ekman mapping to

    Returns:
        mapped_value: int - The mapped value
    """
    emotion_key = list(emotion_mapping.keys())[value]
    for ekman_key, replaced_emotions in ekman_idx_to_emotion_mapping.items():
        if emotion_key in replaced_emotions:
            return emotion_mapping[ekman_key]
    return value


def load_custom_dataset(path):
    """
    Loads our custom joint dataset for eval

    Inputs:
        path: str - The path to the custom dataset

    Returns:
        X: list - The text data
        y_emotion: list - The emotion data
        y_topic: list - The topic data
    """
    with open(path, 'r') as file:
        data = json.load(file)

    X = [obj['text'] for obj in data]
    y_emotion = [obj['emotion'][0] for obj in data]
    y_topic = [obj['topic'] for obj in data]

    return X, y_emotion, y_topic


def preprocess_custom_dataset_logistic_regression(
        X, y_emotion, y_topic, vectorizer,
        emotion_mapping,
        topic_mapping,
        ekman_idx_to_emotion_mapping,
        simplify_with_ekman=False,
        ignore_neutral=False,
    ):
    """
    Preprocesses the custom dataset

    Inputs:
        X: list - The text data to vectorize
        y_emotion: list - The emotion target data
        y_topic: list - The topic target data
        vectorizer: TfidfVectorizer object
        emotion_mapping: dict - The emotion mapping
        topic_mapping: dict - The topic mapping
        ekman_idx_to_emotion_mapping: dict - The Ekman index to emotion mapping
        simplify_with_ekman: boolean indicating whether to simplify the classes with provided Ekman mapping

    Returns:
        X: list - The vectorized text data
        y_emotion: list - The emotion target data
        y_topic: list - The topic target data
    """
    y_emotion = [emotion_mapping[emotion] for emotion in y_emotion]
    y_topic = [topic_mapping[topic] for topic in y_topic]

    X = vectorizer.transform(X)
    if simplify_with_ekman:
        print("\tSimplifying classes...")
        y_emotion = [apply_ekman_mapping(emotion, emotion_mapping, ekman_idx_to_emotion_mapping) for emotion in y_emotion]
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


def preprocess_custom_dataset_llm(    
        X, y_emotion, y_topic,
        emotion_mapping,
        topic_mapping,
        ekman_idx_to_emotion_mapping,
        simplify_with_ekman=False,
        ignore_neutral=False,
    ):
    """
    Preprocesses the custom dataset

    Inputs:
        X: list - The text data to vectorize
        y_emotion: list - The emotion target data
        y_topic: list - The topic target data
        emotion_mapping: dict - The emotion mapping
        topic_mapping: dict - The topic mapping
        ekman_idx_to_emotion_mapping: dict - The Ekman index to emotion mapping
        simplify_with_ekman: boolean indicating whether to simplify the classes with provided Ekman mapping
        ignore_neutral: boolean indicating whether to ignore the neutral class

    Returns:
        X: list - The vectorized text data
        y_emotion: list - The emotion target data
        y_topic: list - The topic target data
    """
    y_emotion = [emotion_mapping[emotion] for emotion in y_emotion]
    y_topic = [topic_mapping[topic] for topic in y_topic]
    X = np.array(X)

    if simplify_with_ekman:
        print("\tSimplifying classes...")
        y_emotion = [apply_ekman_mapping(emotion, emotion_mapping, ekman_idx_to_emotion_mapping) for emotion in y_emotion]
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


def load_custom_data(path, emotion_mapping, topic_mapping, ekman_idx_to_emotion_mapping, vectorizer=None, simplify_with_ekman=False, ignore_neutral=False):
    """
    Load the custom dataset and preprocess it

    Inputs:
        path: str - The path to the custom dataset
        emotion_mapping: dict - The emotion mapping
        topic_mapping: dict - The topic mapping
        ekman_idx_to_emotion_mapping: dict - The Ekman index to emotion mapping
        vectorizer: TfidfVectorizer object - The vectorizer to use for the custom dataset, optional, only for logistic regression
        simplify_with_ekman: boolean indicating whether to simplify the goemotion classes with provided Ekman mapping
        ignore_neutral: boolean indicating whether to ignore the neutral class

    Returns:
        X: numpy array containing the preprocessed text data
        y_emotion: numpy array containing the preprocessed emotion data
        y_topic: numpy array containing the preprocessed topic data
    """
    print(f"Loading custom dataset {path}...\n")
    X, y_emotion, y_topic = load_custom_dataset(path)
    print("Preprocessing custom dataset...")
    if vectorizer:
        X, y_emotion, y_topic = preprocess_custom_dataset_logistic_regression(X, y_emotion, y_topic, \
                                                        vectorizer, emotion_mapping, topic_mapping, \
                                                        ekman_idx_to_emotion_mapping, \
                                                        simplify_with_ekman, \
                                                        ignore_neutral \
                                                        )
    else:
        X, y_emotion, y_topic = preprocess_custom_dataset_llm(X, y_emotion, y_topic, \
                                                        emotion_mapping, topic_mapping, \
                                                        ekman_idx_to_emotion_mapping, \
                                                        simplify_with_ekman, \
                                                        ignore_neutral \
                                                        )

    return X, y_emotion, y_topic
