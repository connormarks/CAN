# How to set up Ollama for generation

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
1. Make sure Ollama is running on your machine
2. Install dependencies
    1. (Optional) run `python -m venv venv` to create a venv. Then source it with `source venv/bin/activarte` (MacOS)
    2. Install requirements with `pip install -r requirements.txt`
3. Run `generate.py`, walk through the questions to select your model and generate examples
4. Run `verify.py` to verify each generated example.
5. Find your synthetic data in Output/Processed
    - Verified examples will have `"verified": true`

## Findings
Findings through testing are recorded in [Findings.md](Findings.md)