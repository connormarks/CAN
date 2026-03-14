# Authors: Nathan Pietrantonio
from .gemini_api import get_models as get_gemini_models
from .ollama_api import get_models as get_ollama_models

def select_model(api_models=False):
    """
    Prompt the user to select a model from the available models on their machine.
    Returns the selected model name.

    Returns:
        model: str - The selected model name.
    """
    if api_models:
        models = get_gemini_models()
        alt_choice = "Use Ollama"
    else:
        models = get_ollama_models()
        alt_choice = "Use Gemini API"

    print()
    print("Available models:")
    for index, model in enumerate(models):
        print(f'\t {index}. {model}')
    print(f'\t {len(models)}. {alt_choice}')
    print()
    choice = input("Enter the model you want to use: ")
    try:
        choice = int(choice)
        if choice == len(models):
            return select_model(api_models=not api_models)
        model = models[int(choice)]
    except Exception as e:
        print(f"Invalid choice: {e}")
        return select_model(api_models)

    print(f"Selected model: {model}")
    print()
    return api_models, model
