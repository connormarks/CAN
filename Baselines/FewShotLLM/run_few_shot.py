from generate import select_model

def _get_run_method():
    if local_or_api == "local":
        return 'local'
    elif local_or_api == "api":
        return 'api'
    else:
        print("Invalid input")
        return _get_run_method()

if __name__ == "__main__":
    run_method = _get_run_method()
    model = select_model(api_models=run_method == 'api')
    print(f"Selected model: {model}")
