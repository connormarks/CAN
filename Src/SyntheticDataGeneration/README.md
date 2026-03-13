This code was created by Nathan Pietrantonio

# How to set up Ollama for generation

## Local vs API
This can be run locally (Ollama) or using the free tier on the Gemini API (recommended). Skip to "Using our data generation tools" if using the API.

## Install Ollama
Install Ollama for your platform on their website:
[Download Ollama](https://ollama.com/download/mac)

## Run Ollama
Run Ollama in the terminal. To verify installation, run `ollama --version`.
You may need to create an account on the [Ollama website](https://ollama.com/) and sign in using the command line tool.

## Install models
Install a model which is capable enough to generate synthetic data. For testing, we used `mistral:7b-instruct`.

To install a model, run `ollama run <model name>`
This will download the model, then run it to show that it is working.

## Using our data generation tools
Please use Python version `3.12`
### Local generation (local, lower quality)
1. Make sure Ollama is running on your machine
2. Install dependencies
    1. (Optional) run `python -m venv venv` to create a venv. Then source it with `source venv/bin/activarte` (MacOS)
    2. Install requirements with `pip install -r requirements.txt`
3. Run `generate.py`, walk through the questions to select your model and generate examples
4. Run `verify.py` to verify each generated example.
5. Find your synthetic data in `Output/Processed`
    - Verified examples will have `"verified": true`
6. Find your merged dataset (if you elected to merge) in `Output/Processed/Merged/merged_data.json`

### API Generation (higher quality)
1. Log into Google AI Studio and setup an API key
2. Add the API key to a `.env` file. See `.env.example` for an example
3. Install dependencies
    1. (Optional) run `python -m venv venv` to create a venv. Then source it with `source venv/bin/activarte` (MacOS)
    2. Install requirements with `pip install -r requirements.txt`
4. Run `generate.py` to manually create and verify examples OR run `auto_generate.py` to automatically create a joint output file (recommended)
5. Find your synthetic data in `Output/Processed/Merged/merged_data.json`
6. (Optional) run `show_stats.py` to show stats on class counts and joint class counts (emotion n about topic m)
    - Output files with these counts will be in `Output/Processed/Merged`

## Findings
Findings through testing are recorded in [Findings.md](Findings.md)