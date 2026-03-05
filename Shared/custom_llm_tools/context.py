from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions


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
