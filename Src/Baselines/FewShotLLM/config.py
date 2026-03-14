# Output directories and class mappings
# Authors: Nathan Pietrantonio
MERGED_DATASET_PATH = "../../SyntheticDataGeneration/Output/Merged/merged_data.json"
"""Location of merged custom dataset"""
MODEL_PATH = "models"
"""Folder to save the trained models"""

EMOTION_MAPPING = {
    "admiration": 0,
    "amusement": 1,
    "anger": 2,
    "annoyance": 3,
    "approval": 4,
    "caring": 5,
    "confusion": 6,
    "curiosity": 7,
    "desire": 8,
    "disappointment": 9,
    "disapproval": 10,
    "disgust": 11,
    "embarrassment": 12,
    "excitement": 13,
    "fear": 14,
    "gratitude": 15,
    "grief": 16,
    "joy": 17,
    "love": 18,
    "nervousness": 19   ,
    "optimism": 20,
    "pride": 21,
    "realization": 22,
    "relief": 23,
    "remorse": 24,
    "sadness": 25,
    "surprise": 26,
    "neutral": 27,
}
"""Mapping of emotion class to index"""


EKMAN_IDX_TO_EMOTION_MAPPING = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude",  "love", "optimism", "relief", "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief",  "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"]
}
"""Mapping of simplified emotion class to original emotion classes"""


TOPIC_MAPPING = {
    "World": 1,
    "Sports": 2,
    "Business": 3,
    "Sci/Tech": 4,
}
"""
Mapping of topic class to index, 1 indexed in within the dataset
Mapping from https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset
"""
