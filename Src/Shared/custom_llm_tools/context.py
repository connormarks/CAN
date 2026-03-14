# Authors: Nathan Pietrantonio
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
import json
import string


def _setup_context_for_gemini(prompt, system=None, message_history={}):
    prior_messages = message_history.get("contents", [])
    messages = prior_messages + [{'role': 'user', 'parts': [{'text': prompt}]}]
    if system:
        system = GenerateContentConfig(
                    system_instruction=system.split('\n')
                )
    else:
        system = None
    return system, {"contents": messages}


def _setup_context_for_ollama(prompt, system=None, message_history=[]):
    messages = message_history + [{'role': 'user', 'content': prompt}]
    if system:
        messages.insert(0, {'role': 'system', 'content': system})
    return messages


def _add_new_message_for_gemini(message_history, message):
    return {"contents": message_history + [{'role': 'assistant', 'parts': [{'text': message}]}]}


def _add_new_message_for_ollama(message_history, message):
    return message_history + [{'role': 'assistant', 'content': message}]


def setup_context(prompt, system=None, message_history=None, for_api=False):
    """
    Set up message history for the LLM

    Input:
        prompt: str - The prompt to the LLM.
        system: str - The system prompt as a string.
        message_history: list - The message history as a list of dictionaries.
        for_api: bool - Whether to use the API context setup.
    Returns:
        system: GenerateContentConfig | str | None - The system configuration for the Gemini API or the string for Ollama.
        messages: list - The messages as a list of dictionaries.
    """
    if for_api:
        return _setup_context_for_gemini(prompt, system, message_history or {})
    else:
        return system, _setup_context_for_ollama(prompt, system, message_history or [])


def add_new_message(message_history, message, for_api=False):
    """
    Add a new message to the message history.

    Input:
        message_history: list - The message history as a list of dictionaries.
        message: str - The message to add to the message history.
    Returns:
        message_history: list - The message history as a list of dictionaries.
    """
    if for_api:
        return _add_new_message_for_gemini(message_history, message)
    else:
        return _add_new_message_for_ollama(message_history, message)


def load_system_prompt(file_path, template_filler={}):
    """
    Load the system prompt from the file path.
    If a template filler is provided, it will be used to fill in the template.
    Returns the system prompt as a string.

    Input:
        file_path: str - The path to the system prompt file.
        template_filler: dict - The template filler as a dictionary. Defaults to an empty dictionary.
    Returns:
        system_prompt: str - The system prompt as a string.
    """
    with open(file_path, 'r') as file:
        system_prompt = file.read()
        if template_filler:
            system_prompt = string.Template(system_prompt)
            system_prompt = system_prompt.substitute(template_filler)
        return system_prompt


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
    print(cleaned_json_string)
    json_obj = json.loads(cleaned_json_string)
    # Add verified flag to each object
    for obj in json_obj:
        obj['verified'] = False
        obj['needs_editing'] = None
    return json_obj
