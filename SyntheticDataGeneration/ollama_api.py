from context import setup_context, add_new_message
import ollama


def get_models():
    """
    Get the list of models available on the machine.

    Returns:
        models: list - The list of models as a list of strings.
    """
    models = ollama.list().models
    return [model.model for model in models]


def stream_response(model, prompt, system=None, message_history=[]):
    """
    Stream the response from the LLM.
    This function outputs the response to the console in addition to
    building a message history.

    Input:
        model: str - The model to use.
        prompt: str - The prompt to the LLM.
        system: str - The system prompt as a string.
        message_history: list - The message history as a list of dictionaries.
    Returns:
        new_message_history: list - The message history as a list of dictionaries.
    """
    _, messages = setup_context(prompt, system, message_history)
    stream = ollama.chat(model=model, messages=messages, stream=True)

    message = ''
    for chunk in stream:
        message += chunk['message']['content']
        print(chunk['message']['content'], end='', flush=True)

    new_message_history = add_new_message(message_history, message)
    return new_message_history


def generate_response(model, prompt, system=None, message_history=[]):
    """
    Generate the response from the LLM.

    Input:
        model: str - The model to use.
        prompt: str - The prompt to the LLM.
        system: str - The system prompt as a string.
        message_history: list - The message history as a list of dictionaries.
    Returns:
        new_message_history: list - The message history as a list of dictionaries.
    """
    _, messages = setup_context(prompt, system, message_history)
    response = ollama.chat(model=model, messages=messages)
    new_message_history = add_new_message(message_history, response['message']['content'])
    return new_message_history
