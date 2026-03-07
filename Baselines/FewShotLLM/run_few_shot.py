from custom_llm_tools.tools import select_model
from custom_llm_tools.custom_data import apply_ekman_mapping, load_custom_data, load_custom_data
from config import EMOTION_MAPPING, TOPIC_MAPPING, EKMAN_IDX_TO_EMOTION_MAPPING
import json
import os

DATA_PATH = '../../SyntheticDataGeneration/Output/Merged/merged_data.json'

def _get_data_path(data_path=DATA_PATH):
    """
    Get the data path from the default data path, or prompt the user for a different path.
    Returns the data path.
    """
    # Check if the file exists
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        print(f"File does not exist at {data_path}")
        data_path = input("Enter the path to the data file: ")
        return _get_data_path(data_path)
    
    return data_path


if __name__ == "__main__":
    is_api, model = select_model()
    data_path = _get_data_path()
    print()

    apply_ekman = input("Apply Ekman mapping? (y/n): ") == "y"
    remove_neutral = input("Remove neutral class? (y/n): ") == "y"
    print()

    X, y_emotion, y_topic = load_custom_data(data_path, EMOTION_MAPPING, TOPIC_MAPPING, EKMAN_IDX_TO_EMOTION_MAPPING, 
                                   simplify_with_ekman=apply_ekman, ignore_neutral=remove_neutral)


    print(f"Loaded {len(X)} examples\n")
