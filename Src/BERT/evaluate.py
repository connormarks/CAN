# Authors: Connor Marks, Avery Horton


import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
from matplotlib.colors import LogNorm
import os


TOPIC_MAPPING = { #topics
    "World": 0,
    "Sports": 1,
    "Business": 2,
    "Sci/Tech": 3,
    "NULL": -100
}
REVERSE_TOPIC_MAPPING = {v: k for k, v in TOPIC_MAPPING.items()}

EMOTION_CATEGORIES = [ #ekman
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "neutral"
]


def evaluate(model, loader, device, run_dir, epoch):
    model.eval()

    topic_true = []
    topic_pred = []

    emotion_true = []
    emotion_pred = []

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
    plt.title(f"Topic Confusion Matrix - Epoch {epoch}") #same as before
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"topic_cm_epoch{epoch}.png"), dpi=300)
    plt.close()

    emotion_true_np = np.array(emotion_true)
    emotion_pred_np = np.array(emotion_pred)

    emotion_micro_f1 = f1_score( #stop metric
        emotion_true_np,
        emotion_pred_np,
        average="micro",
        labels=list(range(len(EMOTION_CATEGORIES))),
        zero_division=0
    )

    emotion_macro_f1 = f1_score(
        emotion_true_np,
        emotion_pred_np,
        average="macro",
        labels=list(range(len(EMOTION_CATEGORIES))),
        zero_division=0
    )

    emotion_per_class_f1 = f1_score(
        emotion_true_np,
        emotion_pred_np,
        average=None,
        labels=list(range(len(EMOTION_CATEGORIES))),
        zero_division=0
    )

    emotion_support = np.bincount(
        emotion_true_np,
        minlength=len(EMOTION_CATEGORIES)
    )

    emotion_cm = confusion_matrix( #new emotion CM for png
        emotion_true_np,
        emotion_pred_np,
        labels=list(range(len(EMOTION_CATEGORIES)))
    )

    emotion_names_with_support = [
        f"{EMOTION_CATEGORIES[i]}({emotion_support[i]})"
        for i in range(len(EMOTION_CATEGORIES))
    ]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        emotion_cm,
        annot=True,
        norm=LogNorm(),
        fmt="d",
        cmap="Blues",
        xticklabels=emotion_names_with_support,
        yticklabels=emotion_names_with_support
    )
    plt.title(f"Emotion Confusion Matrix - Epoch {epoch}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"emotion_cm_epoch{epoch}.png"), dpi=300)
    plt.close()

    model.train()

    topic_accuracy = np.mean(np.array(topic_true) == np.array(topic_pred))

    return {
        "topic_accuracy": topic_accuracy,
        "emotion_micro_f1": emotion_micro_f1,
        "emotion_macro_f1": emotion_macro_f1,
        "emotion_per_class_f1": emotion_per_class_f1.tolist(),
        "emotion_support": emotion_support.tolist()
    }