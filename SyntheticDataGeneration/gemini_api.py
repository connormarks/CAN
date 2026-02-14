"""
This file contains the API for the Gemini model.
Code from example at https://github.com/google-gemini/cookbook/blob/main/quickstarts/Streaming.ipynb
"""

from google import genai
from dotenv import load_dotenv
from context import setup_context, add_new_message
import os


def create_client():
    """
    Create a client for the Gemini API.

    Returns:
        client: genai.Client - The client for the Gemini API.
    """
    load_dotenv()
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    return genai.Client(api_key=gemini_api_key)


def get_models():
    """
    Get the list of models available on the machine.

    Returns:
        models: list - The list of models as a list of strings.
    """
    client = create_client()
    models = client.models.list()
    return [model.name for model in models]


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
    client = create_client()
    system, messages = setup_context(prompt, system, message_history, for_api=True)

    message = ''
    for chunk in client.models.generate_content_stream(
            model=model,
            contents=messages["contents"],
            config=system
        ):
        message += chunk.text
        print(chunk.text, end='', flush=True)

    new_message_history = add_new_message(message_history, message, for_api=True)
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
    client = create_client()
    system, messages = setup_context(prompt, system, message_history, for_api=True)

    response = client.models.generate_content(
        model=model,
        contents=messages["contents"],
        config=system
    )

    new_message_history = add_new_message(message_history, response.text, for_api=True)
    return new_message_history
