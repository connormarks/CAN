from ollama_api import get_models, stream_response
import datetime
import json
from config import OUTPUT_DIR

def select_model():
    """
    Prompt the user to select a model from the available models on their machine.
    Returns the selected model name.

    Returns:
        model: str - The selected model name.
    """
    models = get_models()
    print()
    print("Available models:")
    for index, model in enumerate(models):
        print(f'\t {index}. {model}')
    print()
    choice = input("Enter the model you want to use: ")
    try:
        choice = int(choice)
        model = models[int(choice)]
    except Exception:
        print(f"Invalid choice")
        return select_model()
    print(f"Selected model: {model}")
    print()
    return model


def load_system_prompt(file_path):
    """
    Load the system prompt from the file path.
    Returns the system prompt as a string.

    Input:
        file_path: str - The path to the system prompt file.
    Returns:
        system_prompt: str - The system prompt as a string.
    """
    with open(file_path, 'r') as file:
        return file.read()


def format_json(llm_json_string):
    """
    Format the JSON string returned by the LLM into a list of dictionaries.
    Returns the list of dictionaries and adds a verified flag to each object.

    Input:
        llm_json_string: str - The JSON string returned by the LLM.
    Returns:
        json_obj: list - The list of dictionaries.
    """
    cleaned_json_string = llm_json_string.replace('```json', '').replace('```', '')
    json_obj = json.loads(cleaned_json_string)
    # Add verified flag to each object
    for obj in json_obj:
        obj['verified'] = False
        obj['needs_editing'] = None
    return json_obj


if __name__ == "__main__":
    """
    Main function to generate synthetic data.
    Prompts the user to select a model, enter the number of examples to generate, and save the data.
    """
    system = load_system_prompt('prompt.txt')
    model = select_model()
    num_examples = int(input("Enter the number of examples you want to generate: "))
    print()
    prompt = f"Please generate {num_examples} examples"
    result = stream_response(model, prompt, system)[0]['content']
    print()
    formatted_json = format_json(result)
    save = True if input("Save data? (y/n): ") == 'y' else False
    if save:
        filename = f"generated_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(f'{OUTPUT_DIR}/{filename}', 'w') as file:
            json.dump(formatted_json, file, indent=4)
        print(f"Data saved to {filename}")
    else:
        print("Data not saved")
