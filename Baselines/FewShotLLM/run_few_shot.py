from custom_llm_tools.tools import select_model
import json
import os

DATA_PATH = '../../SyntheticDataGeneration/Output/Merged/merged_data.json'

def _load_data(data_path=DATA_PATH):
    """
    Load the data from the default data path, or prompt the user for a different path.
    Returns the data as a list of dictionaries.
    """
    # Check if the file exists
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        print(f"File does not exist at {data_path}")
        data_path = input("Enter the path to the data file: ")
        return _load_data(data_path)
    # Load the data
    with open(data_path, 'r') as file:
        data = json.load(file)
        print(f"Loaded {len(data)} examples\n")
        return data

if __name__ == "__main__":
    is_api, model = select_model()
    data = _load_data()