from custom_llm_tools.tools import select_model

if __name__ == "__main__":
    is_api, model = select_model()
    print(f"Selected model: {model}")
