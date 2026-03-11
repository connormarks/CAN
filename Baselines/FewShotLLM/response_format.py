import ollama
from pydantic import BaseModel
from typing import Literal

def get_response_format(emotions, topics):
    """
    Get the response format for the model to use.
    Returns the response format as a BaseModel.

    Input:
        emotions: list - The list of emotions to use.
        topics: list - The list of topics to use.
    Returns:
        ResponseFormat: The response format as a BaseModel.
    """
    emotion_literal = Literal[tuple(emotions)]
    topic_literal = Literal[tuple(topics)]

    class ResponseFormat(BaseModel):
        emotion: emotion_literal
        topic: topic_literal

    return ResponseFormat
