import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# used in matrix charting
TOPIC_MAPPING = {
    "World": 0,
    "Sports": 1,
    "Business": 2,
    "Sci/Tech": 3,
    "NULL": -100
}
REVERSE_TOPIC_MAPPING = {v: k for k, v in TOPIC_MAPPING.items()}
EMOTION_MAPPING = {
    "admiration": 0,
    "amusement": 1,
    "anger": 2,
    "annoyance": 3,
    "approval": 4,
    "caring": 5,
    "confusion": 6,
    "curiosity": 7,
    "desire": 8,
    "disappointment": 9,
    "disapproval": 10,
    "disgust": 11,
    "embarrassment": 12,
    "excitement": 13,
    "fear": 14,
    "gratitude": 15,
    "grief": 16,
    "joy": 17,
    "love": 18,
    "nervousness": 19   ,
    "optimism": 20,
    "pride": 21,
    "realization": 22,
    "relief": 23,
    "remorse": 24,
    "sadness": 25,
    "surprise": 26,
    "neutral": 27,
    "NULL": -100
}
REVERSE_EMOTION_MAPPING = {v: k for k, v in EMOTION_MAPPING.items()}
EKMAN_MAPPING = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude",  "love", "optimism", "relief", "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief",  "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"]
}
EKMAN_CATEGORIES = list(EKMAN_MAPPING.keys())

def map_to_ekman(emotion_idx):
    return EKMAN_CATEGORIES[emotion_idx]

def evaluate(model, loader, device, run_dir, epoch):
    model.eval()

    topic_true = []
    topic_pred = []

    emotion_true = [] # leaving these in in case we want all 28 emotions again later
    emotion_pred = []

    ekman_true = []
    ekman_pred = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            emotion_labels = batch["emotion_labels"].to(device)
            topic_labels = batch["topic_label"].to(device)

            logits = model(input_ids, attention_mask)

            emotion_logits = logits["emotion_logits"]
            topic_logits = logits["topic_logits"]

            topic_preds = torch.argmax(topic_logits, dim=1)
            emotion_preds = torch.argmax(emotion_logits, dim=1)

            for i in range(len(topic_labels)):
                if topic_labels[i].item() != -100:
                    topic_true.append(topic_labels[i].item())
                    topic_pred.append(topic_preds[i].item())
                if emotion_labels[i].sum() > 0:
                    true_emotion = torch.argmax(emotion_labels[i]).item()
                    pred_emotion = emotion_preds[i].item()
                    emotion_true.append(true_emotion)
                    emotion_pred.append(pred_emotion)

                    # map to EKMAN for CM output
                    ekman_true.append(map_to_ekman(true_emotion))
                    ekman_pred.append(map_to_ekman(pred_emotion))

    topic_labels_list = sorted([
        v for v in TOPIC_MAPPING.values() if v != -100
    ])
    topic_names = [REVERSE_TOPIC_MAPPING[i] for i in topic_labels_list]

    topic_cm = confusion_matrix(
        topic_true,
        topic_pred,
        labels=topic_labels_list
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        topic_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=topic_names,
        yticklabels=topic_names
    )
    plt.title(f"Topic Confusion Matrix - Epoch {epoch}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"topic_cm_epoch{epoch}.png"), dpi=300)
    plt.close()

    emotion_cm = confusion_matrix(
        ekman_true,
        ekman_pred,
        labels=EKMAN_CATEGORIES
    )

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        emotion_cm,
        annot=False,
        cmap="Blues",
        xticklabels=EKMAN_CATEGORIES,
        yticklabels=EKMAN_CATEGORIES
    )
    plt.title(f"Emotion Confusion Matrix - Epoch {epoch}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"emotion_cm_epoch{epoch}.png"), dpi=300)
    plt.close()

    model.train()