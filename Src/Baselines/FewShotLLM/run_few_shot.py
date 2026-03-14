# Authors: Nathan Pietrantonio
from custom_llm_tools.tools import select_model
from custom_llm_tools.custom_data import apply_ekman_mapping, load_custom_data, load_custom_data
from custom_llm_tools.context import load_system_prompt, add_new_message, format_json
from custom_llm_tools.ollama_api import generate_response
from response_format import get_response_format
from config import EMOTION_MAPPING, TOPIC_MAPPING, EKMAN_IDX_TO_EMOTION_MAPPING
from scoring import custom_scoring, joint_scoring
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


def _classify_example(model, example, system, y_emotion, y_topic, response_format, max_retries=3):
    """
    Classify an example using the model and system.
    Returns the emotion and topic predictions.
    """
    message_history = []
    retry_count = 0
    max_retries = max_retries
    def _classify(_message_history, _retry_count, _max_retries):
        if _retry_count > _max_retries:
            return None, None
        try:
            response = generate_response(model, example, system, _message_history, response_format)
            cleaned_json_string = response[-1]['content'].replace('```json', '').replace('```JSON', '').replace('```', '')
            response_json = json.loads(cleaned_json_string)
        except json.JSONDecodeError:
            print("Invalid JSON, retrying...")
            return _classify(_message_history, _retry_count + 1, _max_retries)
        except Exception as e:
            print(f"Error classifying example: {e}. Retrying...")
            return _classify(_message_history, _retry_count + 1, _max_retries)

        emotion_prediction = response_json['emotion']
        topic_prediction = response_json['topic']
        
        try:
            emotion_class = EMOTION_MAPPING[emotion_prediction]
            topic_class = TOPIC_MAPPING[topic_prediction]
        except KeyError:
            print("Invalid emotion or topic, retrying...")
            return _classify(_message_history, _retry_count + 1, _max_retries)
        return emotion_class, topic_class

    return _classify(message_history, retry_count, max_retries)


def _get_next_output_index():
    """
    Returns the next output index for the output file.
    """
    output_files = os.listdir(OUTPUT_DIR)
    if not output_files:
        return 1
    return len(output_files) + 1


def _get_user_input():
    is_api, model = select_model()
    data_path = _get_data_path()
    print()

    apply_ekman = input("Apply Ekman mapping? (y/n): ") == "y"
    remove_neutral = input("Remove neutral class? (y/n): ") == "y"
    save_every_n_examples = int(input("Save every n examples? (0 to not save): "))
    print()
    return is_api, model, data_path, apply_ekman, remove_neutral, save_every_n_examples


def _run_few_shot(X, y_emotion, y_topic, model, system, response_format, save_every_n_examples, apply_ekman, remove_neutral):
    save_data = {"model": model, "apply_ekman": apply_ekman, 
                "remove_neutral": remove_neutral, "classifications": [], 
                "unable_to_classify": 0, "total_examples": 0}
    save_file_name = f"run_{_get_next_output_index()}.json"
    save_file = f"{OUTPUT_DIR}/{save_file_name}"
    for example, emotion, topic in zip(X, y_emotion, y_topic):
        emotion_class, topic_class = _classify_example(model, example, system, emotion, topic, response_format)

        if emotion_class is None or topic_class is None:
            print("Error classifying example")
            save_data["unable_to_classify"] += 1
            emotion_class = -1
            topic_class = -1

        print(f"Emotion validation: {emotion_class == emotion}\nTopic validation: {topic_class == topic}")
        
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

    return save_file_name


def _show_run_results(run_file):
    with open(f"{OUTPUT_DIR}/{run_file}", 'r') as f:
        run_data = json.load(f)

    inverted_emotion_mapping = {v: k for k, v in EMOTION_MAPPING.items()}
    inverted_topic_mapping = {v: k for k, v in TOPIC_MAPPING.items()}

    examples = [classification['example'] for classification in run_data['classifications']]
    true_emotions = [inverted_emotion_mapping[classification['true_emotion']] for classification in run_data['classifications']]
    predicted_emotions = [inverted_emotion_mapping[classification['predicted_emotion']] for classification in run_data['classifications']]
    true_topics = [inverted_topic_mapping[classification['true_topic']] for classification in run_data['classifications']]
    predicted_topics = [inverted_topic_mapping[classification['predicted_topic']] for classification in run_data['classifications']]

    valid_emotion_count = sum([true_emotion == predicted_emotion for true_emotion, predicted_emotion in zip(true_emotions, predicted_emotions)])
    valid_topic_count = sum([true_topic == predicted_topic for true_topic, predicted_topic in zip(true_topics, predicted_topics)])

    print(f"Loaded {len(run_data['classifications'])} classification results")
    print(f"Unable to classify: {run_data['unable_to_classify']}, {run_data['unable_to_classify'] / len(run_data['classifications']) * 100}%")
    print(f"Emotion accuracy: {valid_emotion_count / len(run_data['classifications']) * 100}%")
    print(f"Topic accuracy: {valid_topic_count / len(run_data['classifications']) * 100}%")
    input()

    custom_scoring(examples, true_emotions, true_topics, predicted_emotions, predicted_topics)
    input()
    joint_scoring(examples, true_emotions, true_topics, predicted_emotions, predicted_topics)
    input()


if __name__ == "__main__":
    load_run = input("Load previous run? (y/n): ") == "y"
    if load_run:
        run_file = input("Enter run file name: ")
        print()
        _show_run_results(run_file)
        exit()

    is_api, model, data_path, apply_ekman, remove_neutral, save_every_n_examples = _get_user_input()

    X, y_emotion, y_topic = load_custom_data(data_path, EMOTION_MAPPING, TOPIC_MAPPING, EKMAN_IDX_TO_EMOTION_MAPPING, 
                                   simplify_with_ekman=apply_ekman, ignore_neutral=remove_neutral)

    print(f"Loaded {len(X)} examples")
    print(f"Emotion classes: {len(set(y_emotion))}")
    print(f"Topic classes: {len(set(y_topic))}")
    print()

    emotion_template = _get_emotion_template(apply_ekman, remove_neutral)
    system = load_system_prompt(SYSTEM_PROMPT_PATH, template_filler={'EMOTIONS': load_system_prompt(emotion_template)})

    # Get the emotion and topic classes from the mapping
    emotions = [emotion for emotion, idx in EMOTION_MAPPING.items() if idx in y_emotion]
    topics = [topic for topic, idx in TOPIC_MAPPING.items() if idx in y_topic]
    # Create a response format for the model to use, restricting the output to the valid options
    response_format = get_response_format(emotions, topics)

    save_file_name = _run_few_shot(X, y_emotion, y_topic, 
                                    model, system, response_format, 
                                    save_every_n_examples, apply_ekman, remove_neutral
                                  )

    _show_run_results(save_file_name)
