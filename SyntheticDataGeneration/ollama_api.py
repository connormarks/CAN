import ollama

def _setup_context(prompt, system=None, message_history=[]):
    messages = message_history + [{'role': 'user', 'content': prompt}]
    if system:
        messages.insert(0, {'role': 'system', 'content': system})
    return messages


def _add_new_message(message_history, message):
    return message_history + [{'role': 'assistant', 'content': message}]


def get_models():
    models = ollama.list().models
    return [model.model for model in models]


def stream_response(model, prompt, system=None, message_history=[]):
    messages = _setup_context(prompt, system, message_history)
    stream = ollama.chat(model=model, messages=messages, stream=True)

    message = ''
    for chunk in stream:
        message += chunk['message']['content']
        print(chunk['message']['content'], end='', flush=True)

    new_message_history = _add_new_message(message_history, message)
    return new_message_history


def generate_response(model, prompt, system=None, message_history=[]):
    messages = _setup_context(prompt, system, message_history)
    response = ollama.chat(model=model, messages=messages)
    new_message_history = _add_new_message(message_history, response['message']['content'])
    return new_message_history
