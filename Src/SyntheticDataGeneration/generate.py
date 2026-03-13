from custom_llm_tools.ollama_api import get_models as get_ollama_models, stream_response as stream_ollama_response
from custom_llm_tools.gemini_api import get_models as get_gemini_models, stream_response as stream_gemini_response
from custom_llm_tools.context import load_system_prompt, format_json
from custom_llm_tools.tools import select_model
from config import OUTPUT_DIR
import datetime
import json

if __name__ == "__main__":
    """
    Main function to generate synthetic data.
    Prompts the user to select a model, enter the number of examples to generate, and save the data.
    """
    print("ERROR")
    print("This file is deprecated. Please use auto_generate.py instead.")
    print("The code is kept for reference.")
    print("Exiting...")
    exit(1)

    system = load_system_prompt('prompt.txt')
    is_api, model = select_model()
    num_examples = int(input("Enter the number of examples you want to generate: "))
    print()
    prompt = f"Please generate {num_examples} examples"

    if is_api:
        result = stream_gemini_response(model, prompt, system)['contents'][0]['parts'][0]['text']
    else:
        result = stream_ollama_response(model, prompt, system)[0]['content']

    print()
    formatted_json = format_json(result)
    save = True if input("Save data? (y/n): ") == 'y' else False
    if save:
        filename = f"{model.strip('models/')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(f'{OUTPUT_DIR}/{filename}', 'w') as file:
            json.dump(formatted_json, file, indent=4)
        print(f"Data saved to {filename}")
    else:
        print("Data not saved")
