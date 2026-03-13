# Authors: Connor Marks

import os
import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer

from CAN.Src.BERT.model import EmotionTopicClassifier
from CAN.Src.BERT import config


TOPICS = {
    "World": 0,
    "Sports": 1,
    "Business": 2,
    "Sci/Tech": 3
}



REV_TOPIC = {v: k for k, v in TOPICS.items()}



EKMAN_MAPPING = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude",  "love", "optimism", "relief", "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief",  "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"]
}

EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]


def map_to_ekman(emotions):
    '''to remap the merged test data from the json into the ekman'''
    v = [0]*len(EMOTIONS) #len 7
    for emo in emotions:
        for i, ekman in enumerate(EMOTIONS):
            if emo in EKMAN_MAPPING[ekman]: #map it
                v[i] = 1
    return v # v is the new true label




def get_names(v):
    '''gives the names of the predictions'''
    return [EMOTIONS[i] for i, value in enumerate(v) if value == 1]


def select_emotion(i):
    '''selects the given index'''
    v = [0] * len(EMOTIONS)
    v[i] = 1
    return v


def main():
    model_path = r"C:\cs175\project\CAN\Src\run\run_9\best.pt" #run goes here
    data_path = r"C:\cs175\project\CAN\Src\SyntheticDataGeneration\Output\Merged\merged_data.json" #this stays fixed
    output_path = r"C:\cs175\project\CAN\Src\test\results.json" #output

    device = torch.device("cuda" if torch.cuda.is_available() and config.USE_CUDA else "cpu") #copied from train
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = EmotionTopicClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Running Now, Please wait...")


    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f) #get the test data

    topic_true = [] 
    topic_pred = []
    emotion_true = []
    emotion_pred = []
    emotion_true_ids = []
    emotion_pred_ids = []
    outputs = []

    with torch.no_grad():
        for row in data: #get each element of a test sample
            text = row["text"]
            true_topic = row["topic"]
            true_emotions_fine = row["emotion"]
            true_emotion_vector = map_to_ekman(true_emotions_fine) #MAPS IT TO OUR EKMAN

            tokens = tokenizer( #now tokenize the text
                text,
                truncation=True,
                padding="max_length",
                max_length=config.MAX_SEQUENCE_LENGTH,
                return_tensors="pt")
            


            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            #forward pass just like in train
            logits = model(input_ids, attention_mask)
            topic_logits = logits["topic_logits"]
            emotion_logits = logits["emotion_logits"]
            
            pred_topic_id = torch.argmax(topic_logits, dim=1).item()#picking the most confident one
            pred_emotion_id = torch.argmax(emotion_logits, dim=1).item()
            true_emotion_id = np.argmax(true_emotion_vector)

            pred_emotion_vector = select_emotion(pred_emotion_id) #make the vector

            #these are just the variables with the inference information
            topic_true.append(TOPICS[true_topic])
            topic_pred.append(pred_topic_id)
            emotion_true.append(true_emotion_vector)
            emotion_pred.append(pred_emotion_vector)
            emotion_true_ids.append(true_emotion_id)
            emotion_pred_ids.append(pred_emotion_id)


            #those go here.
            outputs.append({
                "text": text,
                "true_topic": true_topic,
                "pred_topic": REV_TOPIC[pred_topic_id],
                "true_emotions_fine": true_emotions_fine,
                "true_emotions_ekman": get_names(true_emotion_vector),
                "pred_emotions_ekman": get_names(pred_emotion_vector)})
            

    #calculate accuracy for topic and emotion and f1 for emotion
    topic_accuracy = accuracy_score(topic_true, topic_pred)
    emotion_micro_f1 = f1_score(emotion_true, emotion_pred, average="micro", zero_division=0)
    emotion_macro_f1 = f1_score(emotion_true, emotion_pred, average="macro", zero_division=0)
    emotion_accuracy = accuracy_score(emotion_true_ids, emotion_pred_ids)
    #write results
    results = {
        "metrics": {
            "topic_accuracy": topic_accuracy,
            "emotion_micro_f1": emotion_micro_f1,
            "emotion_macro_f1": emotion_macro_f1,
            "emotion_accuracy": emotion_accuracy
        },
        "predictions": outputs #all of outputs here
    }
    #print it also
    print(f"Topic Accuracy: {topic_accuracy:.6f}")
    print(f"Emotion Micro F1: {emotion_micro_f1:.6f}")
    print(f"Emotion Macro F1: {emotion_macro_f1:.6f}")
    print(f"Emotion Accuracy: {emotion_accuracy:.6f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True) #make the output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False) #write the result

    print(f"saved predictions to {output_path}")


if __name__ == "__main__":
    main()