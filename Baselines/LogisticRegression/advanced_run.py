from tools.dataset import load_datasets
from tools.preprocess import create_vectorizer, preprocess_go_data, preprocess_ag_data
from tools.scoring import default_scoring, custom_scoring, joint_scoring
from custom_llm_tools.custom_data import load_custom_data
from tools.train import get_models
from tools.config import MERGED_DATASET_PATH, EMOTION_MAPPING, TOPIC_MAPPING, EKMAN_IDX_TO_EMOTION_MAPPING
from collections import namedtuple


Data = namedtuple('Data', ['X_train', 'X_test', 'y_train', 'y_test'])
"""
Namedtuple containing the training and test data for a given dataset
"""

def process_data(go_data, ag_data, vectorizer, simplify_with_ekman=False, ignore_neutral=False):
    """
    Process the data for model training

    Inputs:
        go_data: pandas DataFrame containing the goemotion data
        ag_data: pandas DataFrame containing the agnews data
        vectorizer: TfidfVectorizer object
        simplify_with_ekman: boolean indicating whether to simplify the goemotion classes with provided Ekman mapping
        ignore_neutral: boolean indicating whether to ignore the neutral class

    Returns:
        GoData: Data namedtuple containing the training and test data for the goemotion model
        AgData: Data namedtuple containing the training and test data for the agnews model
    """
    print("Preprocessing goemotion data...")
    go_X_train, go_X_test, go_y_train, go_y_test = preprocess_go_data(go_data, vectorizer, 
                                                                       simplify_with_ekman=simplify_with_ekman,
                                                                       fix_class_imbalance=True,
                                                                       ignore_neutral=ignore_neutral)
    print("Preprocessing agnews data...")
    ag_X_train, ag_X_test, ag_y_train, ag_y_test = preprocess_ag_data(ag_data, vectorizer)

    GoData = Data(go_X_train, go_X_test, go_y_train, go_y_test)
    AgData = Data(ag_X_train, ag_X_test, ag_y_train, ag_y_test)

    return GoData, AgData


# def load_custom_data(vectorizer, simplify_with_ekman=False, ignore_neutral=False):
#     """
#     Load the custom dataset and preprocess it

#     Inputs:
#         vectorizer: TfidfVectorizer object
#         simplify_with_ekman: boolean indicating whether to simplify the goemotion classes with provided Ekman mapping
#         ignore_neutral: boolean indicating whether to ignore the neutral class

#     Returns:
#         X: numpy array containing the preprocessed text data
#         y_emotion: numpy array containing the preprocessed emotion data
#         y_topic: numpy array containing the preprocessed topic data
#     """
#     print(f"Loading custom dataset {MERGED_DATASET_PATH}...\n")
#     X, y_emotion, y_topic = load_custom_dataset(MERGED_DATASET_PATH)
#     print("Preprocessing custom dataset...")
#     X, y_emotion, y_topic = preprocess_custom_dataset(X, y_emotion, y_topic, vectorizer, simplify_with_ekman, ignore_neutral)
#     return X, y_emotion, y_topic


if __name__ == "__main__":
    # Load the default datasets
    go_data, ag_data = load_datasets()

    # Create a vectorizer used for all datasets and models
    # https://datascience.stackexchange.com/questions/122056/logisticregression-loading-problem
    texts = [*go_data["text"], *ag_data["Description"]]
    vectorizer = create_vectorizer(texts, save_name="vectorizer")

    # Ask the user if they want to simplify the GoEmotion classes with the provided Ekman mapping
    simplify_with_ekman = input("Simplify GoEmotion classes with Ekman? (y/n): ") == "y"
    ignore_neutral = input("Ignore the neutral class? (y/n): ") == "y"
    print()

    # Process the data for model training
    GoData, AgData = process_data(go_data, ag_data, vectorizer, simplify_with_ekman, ignore_neutral)

    # Load the custom dataset
    X, y_emotion, y_topic = load_custom_data(vectorizer, MERGED_DATASET_PATH, 
                                             EMOTION_MAPPING, TOPIC_MAPPING, \
                                             EKMAN_IDX_TO_EMOTION_MAPPING, \
                                             simplify_with_ekman, ignore_neutral \
                                             )

    # Get the models, either loading from a file or training them
    go_model, ag_model = get_models(GoData.X_train, GoData.y_train, AgData.X_train, AgData.y_train)

    # Score the default dataset
    input("Press Enter to score the default dataset...")
    default_scoring(go_model, ag_model, GoData, AgData)

    # Score the custom dataset
    input("Press Enter to score the custom dataset...")
    custom_scoring(go_model, ag_model, X, y_emotion, y_topic)

    input("Press Enter to score the joint dataset...")
    joint_scoring(go_model, ag_model, X, y_emotion, y_topic, ignore_neutral=ignore_neutral)
