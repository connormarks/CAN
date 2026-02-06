from ollama_api import get_models, stream_response
import json

def select_model():
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
    with open(file_path, 'r') as file:
        return file.read()


def format_json(llm_json_string):
    cleaned_json_string = llm_json_string.replace('```json', '').replace('```', '')
    return json.loads(cleaned_json_string)


if __name__ == "__main__":
    system = load_system_prompt('../prompt.txt')
    model = select_model()
    num_examples = int(input("Enter the number of examples you want to generate: "))
    print()
    prompt = f"Please generate {num_examples} examples"
    result = stream_response(model, prompt, system)[0]['content']
    formatted_json = format_json(result)
    print(formatted_json)
