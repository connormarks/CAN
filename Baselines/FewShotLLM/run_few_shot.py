from custom_llm_tools.tools import select_model
from custom_llm_tools.custom_data import apply_ekman_mapping, load_custom_data, load_custom_data
from custom_llm_tools.context import load_system_prompt, add_new_message, format_json
from custom_llm_tools.ollama_api import generate_response
from config import EMOTION_MAPPING, TOPIC_MAPPING, EKMAN_IDX_TO_EMOTION_MAPPING
import json
import os
import numpy as np

DATA_PATH = '../../SyntheticDataGeneration/Output/Merged/merged_data.json'
SYSTEM_PROMPT_PATH = 'prompts/prompt_template.txt'
OUTPUT_DIR = 'Output'


EMOTION_TEMPLATE_PATHS = {
    'all_emotions': 'prompts/all_emotions.txt',
    'ekman_neutral': 'prompts/7_emotions.txt',
    'ekman_no_neutral': 'prompts/6_emotions.txt',
    'no_ekman_no_neutral': 'prompts/27_emotions.txt',
}
"""Mapping of emotion template path to template name"""


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


def _get_emotion_template(apply_ekman, ignore_neutral):
    """
    Get the emotion template based on the apply_ekman and ignore_neutral flags.
    Returns the emotion template.
    """
    if apply_ekman and ignore_neutral:
        return EMOTION_TEMPLATE_PATHS['ekman_no_neutral']
    elif apply_ekman:
        return EMOTION_TEMPLATE_PATHS['ekman_neutral']
    elif ignore_neutral:
        return EMOTION_TEMPLATE_PATHS['no_ekman_no_neutral']
    else:
        return EMOTION_TEMPLATE_PATHS['all_emotions']


def _classify_example(model, example, system, y_emotion, y_topic, retry_count=0, max_retries=3):
    """
    Classify an example using the model and system.
    Returns the emotion and topic predictions.
    """
    if retry_count > max_retries:
        return None, None
    
    try:
        response = generate_response(model, example, system)[-1]['content']
        cleaned_json_string = response.replace('```json', '').replace('```JSON', '').replace('```', '')
        response_json = json.loads(cleaned_json_string)
    except json.JSONDecodeError:
        print("Invalid JSON, retrying...")
        return _classify_example(model, example, system, y_emotion, y_topic, retry_count + 1, max_retries)
    except Exception as e:
        print(f"Error classifying example: {e}. Retrying...")
        return _classify_example(model, example, system, y_emotion, y_topic, retry_count + 1, max_retries)

    emotion_prediction = response_json['emotion']
    topic_prediction = response_json['topic']

    try:
        emotion_class = EMOTION_MAPPING[emotion_prediction]
        topic_class = TOPIC_MAPPING[topic_prediction]
    except KeyError:
        print("Invalid emotion or topic, retrying...")
        return _classify_example(model, example, system, y_emotion, y_topic, retry_count + 1, max_retries)

    return emotion_class, topic_class


def _get_next_output_index():
    output_files = os.listdir(OUTPUT_DIR)
    if not output_files:
        return 1
    return len(output_files) + 1


if __name__ == "__main__":
    is_api, model = select_model()
    data_path = _get_data_path()
    print()

    apply_ekman = input("Apply Ekman mapping? (y/n): ") == "y"
    remove_neutral = input("Remove neutral class? (y/n): ") == "y"
    save_every_n_examples = int(input("Save every n examples? (0 to not save): "))
    print()

    X, y_emotion, y_topic = load_custom_data(data_path, EMOTION_MAPPING, TOPIC_MAPPING, EKMAN_IDX_TO_EMOTION_MAPPING, 
                                   simplify_with_ekman=apply_ekman, ignore_neutral=remove_neutral)

    print(f"Loaded {len(X)} examples")
    print(f"Emotion classes: {len(set(y_emotion))}")
    print(f"Topic classes: {len(set(y_topic))}")
    print()

    emotion_template = _get_emotion_template(apply_ekman, remove_neutral)
    system = load_system_prompt(SYSTEM_PROMPT_PATH, template_filler={'EMOTIONS': load_system_prompt(emotion_template)})

    save_data = {"model": model, "apply_ekman": apply_ekman, 
                 "remove_neutral": remove_neutral, "classifications": [], 
                 "unable_to_classify": 0, "total_examples": 0}
    save_file = f"{OUTPUT_DIR}/run_{_get_next_output_index()}.json"


    for example, emotion, topic in zip(X, y_emotion, y_topic):
        emotion_class, topic_class = _classify_example(model, example, system, emotion, topic)
        print(f"Emotion validation: {emotion_class == emotion}\nTopic validation: {topic_class == topic}")
        
        if emotion_class is None or topic_class is None:
            print("Error classifying example")
            save_data["unable_to_classify"] += 1
            emotion_class = -1
            topic_class = -1
        
        save_data["total_examples"] += 1
        save_data["classifications"].append({
            "example": example,
            "true_emotion": emotion,
            "true_topic": topic,
            "predicted_emotion": emotion_class,
            "predicted_topic": topic_class,
        })
        if save_every_n_examples > 0 and len(save_data["classifications"]) % save_every_n_examples == 0:
            with open(save_file, 'w') as f:
                json.dump(save_data, f, indent=4)
            print(f"Saved data to {save_file}")
        print()
    
    with open(save_file, 'w') as f:
        json.dump(save_data, f, indent=4)
    print(f"Saved data to {save_file}")
