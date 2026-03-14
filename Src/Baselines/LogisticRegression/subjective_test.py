# Run a subjective test of the baseline with a custom input
# Authors: Nathan Pietrantonio
from tools.config import MODEL_PATH, EMOTION_MAPPING, TOPIC_MAPPING, EKMAN_IDX_TO_EMOTION_MAPPING
from tools.preprocess import create_vectorizer
import pickle


# When trained with Ekman, model outputs EMOTION_MAPPING indices (2, 11, 14, 17, 25, 26, 27), not 0-6
EKMAN_INDEX_TO_NAME = {EMOTION_MAPPING[ek]: ek for ek in EKMAN_IDX_TO_EMOTION_MAPPING}


if __name__ == "__main__":
    go_emotions_model = pickle.load(open(f'{MODEL_PATH}/go_model.pkl', "rb"))
    ag_news_model = pickle.load(open(f'{MODEL_PATH}/ag_model.pkl', "rb"))
    vectorizer = pickle.load(open(f'{MODEL_PATH}/vectorizer.pkl', "rb"))

    n_classes = len(go_emotions_model.classes_)
    print(f"Loaded GoEmotion model has {n_classes} classes (7 = Ekman, 28 = full).")
    simplify_with_ekman = input("Simplify GoEmotion classes with Ekman? (y/n): ") == "y"

    while True:
        text = input("Enter a text to classify, or type 'exit' to quit: ")
        if text == "exit":
            break
        text = [text]
        text = vectorizer.transform(text)
        go_prediction = go_emotions_model.predict(text)[0]
        ag_prediction = ag_news_model.predict(text)[0]

        if simplify_with_ekman:
            go_prediction = EKMAN_INDEX_TO_NAME[go_prediction]
        else:
            go_prediction = list(EMOTION_MAPPING.keys())[go_prediction]
        ag_prediction = list(TOPIC_MAPPING.keys())[ag_prediction-1]

        print(f"GoEmotion prediction: {go_prediction}")
        print(f"AgNews prediction: {ag_prediction}")
        print()
