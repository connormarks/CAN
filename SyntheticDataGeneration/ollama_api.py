import ollama

def _setup_context(prompt, system=None, message_history=[]):
    """
    Set up message history for the LLM

    Input:
        prompt: str - The prompt to the LLM.
        system: str - The system prompt as a string.
        message_history: list - The message history as a list of dictionaries.
    Returns:
        messages: list - The messages as a list of dictionaries.
    """
    messages = message_history + [{'role': 'user', 'content': prompt}]
    if system:
        messages.insert(0, {'role': 'system', 'content': system})
    return messages


def _add_new_message(message_history, message):
    """
    Add a new message to the message history.

    Input:
        message_history: list - The message history as a list of dictionaries.
        message: str - The message to add to the message history.
    Returns:
        message_history: list - The message history as a list of dictionaries.
    """
    return message_history + [{'role': 'assistant', 'content': message}]


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
    messages = _setup_context(prompt, system, message_history)
    stream = ollama.chat(model=model, messages=messages, stream=True)

    message = ''
    for chunk in stream:
        message += chunk['message']['content']
        print(chunk['message']['content'], end='', flush=True)

    new_message_history = _add_new_message(message_history, message)
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
    messages = _setup_context(prompt, system, message_history)
    response = ollama.chat(model=model, messages=messages)
    new_message_history = _add_new_message(message_history, response['message']['content'])
    return new_message_history
