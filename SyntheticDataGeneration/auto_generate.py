from generate import select_model, format_json, load_system_prompt
from config import OUTPUT_DIR, VERIFIED_DIR, MERGED_DIR, EMOTION_LABELS, TOPIC_LABELS
from verify import merge_verified
from gemini_api import generate_response as generate_gemini_response
from ollama_api import generate_response as generate_ollama_response
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google.genai.errors import ServerError
import numpy as np
import time
import datetime
import json


SIMILARITY_THRESHOLD = 0.80
class RateLimitResponse(): pass


def _sleep_with_print(seconds):
    for i in range(seconds):
        print(f"\rSleeping for {seconds - i} seconds... ", end='', flush=True)
        time.sleep(1)
    print()


def _generate_rate_limited_gemini_response(model, prompt, system, tries=0, max_tries=3, sleep_time=10):
    if tries >= max_tries:
        print("Max tries reached.")
        return RateLimitResponse()
    try:
        result = generate_gemini_response(model, prompt, system)['contents'][0]['parts'][0]['text']
    except ServerError:
        print(f"Error generating response. Retrying. {max_tries - tries} tries remaining.")
        _sleep_with_print(sleep_time)
        print("Retrying...", end='', flush=True)
        return _generate_rate_limited_gemini_response(model, prompt, system, tries + 1, max_tries, sleep_time)
    return result


def generate_batch(model, prompt, system, is_api):
    if is_api:
        result = _generate_rate_limited_gemini_response(model, prompt, system)
    else:
        result = generate_ollama_response(model, prompt, system)[0]['content']
    return result


def auto_generate(example_goal):
    system = load_system_prompt('prompt.txt')
    is_api, model = select_model()
    prompt = f"Please generate {example_goal} examples"

    generated = 0
    batch_files = []
    while generated < example_goal:
        print(f"Generating batch {len(batch_files) + 1}... ", end='', flush=True)
        result = generate_batch(model, prompt, system, is_api)

        if isinstance(result, RateLimitResponse):
            print("Rate limited. Stopping generation.")
            break

        formatted_json = format_json(result)
        generated += len(formatted_json)
        
        filename = f"auto_{datetime.datetime.now().strftime('%H%M%S')}_{len(batch_files)}.json"
        with open(f'{OUTPUT_DIR}/{filename}', 'w') as file:
            json.dump(formatted_json, file, indent=4)
        batch_files.append(filename)
        print(f"Batch saved. {generated} examples generated so far.")

    print()
    print(f"Total {generated} examples generated.\n")
    return batch_files


def check_in_corpus(new_embedding, corpus_embeddings):
    if corpus_embeddings.size == 0:
        return False
    
    new_embedding = new_embedding.reshape(1, -1)
    corpus_embeddings = corpus_embeddings.reshape(len(corpus_embeddings), -1)
    cosine_scores = cosine_similarity(new_embedding, corpus_embeddings)[0]
    max_score = cosine_scores.max()
    if max_score > SIMILARITY_THRESHOLD:
        return True
    return False


def verify_batch(vectorizer, corpus, filename):
    with open(f'{OUTPUT_DIR}/{filename}', 'r') as f:
        json_obj = json.load(f)

    for obj in json_obj:
        valid = True
        for emotion in obj['emotion']:
            if emotion not in EMOTION_LABELS:
                obj['verified'] = False
                obj['needs_editing'] = None
                valid = False
                break
        if not valid:
            continue
        if obj['topic'] not in TOPIC_LABELS:
            obj['verified'] = False
            obj['needs_editing'] = None
            continue

        new_embedding = vectorizer.encode([obj['text']])
        if not check_in_corpus(new_embedding, corpus):
            if corpus.size == 0:
                corpus = new_embedding.copy()
            else:
                corpus = np.vstack([corpus, new_embedding])
            obj['verified'] = True
            obj['needs_editing'] = False
        else:
            obj['verified'] = False
            obj['needs_editing'] = False

    with open(f'{VERIFIED_DIR}/{filename}', 'w') as f:
        json.dump(json_obj, f, indent=4)
    print(f"Verified {len(json_obj)} examples from {filename}.")

    return corpus


def verify_generated(batch_files):
    print("Verifying data...")
    vectorizer = SentenceTransformer('all-MiniLM-L6-v2')
    corpus = np.array([])

    for file in batch_files:
        corpus = verify_batch(vectorizer, corpus, file)

    print()
    print(f"Total {len(corpus)} examples added to corpus.")


if __name__ == "__main__":
    # example_goal = int(input("Enter the number of examples you want to generate: "))
    # print()

    # batch_files = auto_generate(example_goal)

    batch_files = [
        'auto_222935_0.json',
        'auto_223616_0.json',
        'auto_223716_1.json',
        'auto_223823_2.json',
        'auto_223919_3.json',
        'auto_224037_4.json',
        'auto_224138_5.json',
        'auto_224234_6.json',
        'auto_224445_7.json'
    ]

    verify_generated(batch_files)

    print("Merging verified data...")
    merge_verified()
    print()
    print(f"Merged data saved to {MERGED_DIR}/merged_data.json")
