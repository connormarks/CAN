import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
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
    "nervousness": 19,
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
            emotion_preds = (torch.sigmoid(emotion_logits) > 0.5).int() #multi-label prediction

            for i in range(len(topic_labels)):
                if topic_labels[i].item() != -100:
                    topic_true.append(topic_labels[i].item())
                    topic_pred.append(topic_preds[i].item())

                if emotion_labels[i].sum() > 0:
                    emotion_true.append(emotion_labels[i].cpu().numpy())# label vector
                    emotion_pred.append(emotion_preds[i].cpu().numpy()) # prediction vector

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

    emotion_true_np = np.array(emotion_true)
    emotion_pred_np = np.array(emotion_pred)

    emotion_micro_f1 = f1_score( 
        emotion_true_np,
        emotion_pred_np,
        average="micro"
    )

    emotion_per_class_f1 = f1_score(
        emotion_true_np,
        emotion_pred_np,
        average=None
    )

    emotion_macro_f1 = f1_score(
        emotion_true_np,
        emotion_pred_np,
        average="macro"
    )
    emotion_support = emotion_true_np.sum(axis=0).astype(int) #how many val samples contained the emotion
    emotion_ids = sorted([v for v in EMOTION_MAPPING.values() if v != -100]) #names
    emotion_names = [f"{REVERSE_EMOTION_MAPPING[i]}({emotion_support[i]})" for i in emotion_ids]

    plt.figure(figsize=(16, 3))#heatmap
    sns.heatmap(
        emotion_per_class_f1.reshape(1, -1),#reshape for heatmap
        annot=False,
        cmap="Blues",
        xticklabels=emotion_names,
        yticklabels=["F1"]
    )
    plt.title(f"Emotion Per-Class F1 - Epoch {epoch}")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"emotion_f1_epoch{epoch}.png"), dpi=300)
    plt.close()

    model.train()

    topic_accuracy = np.mean(np.array(topic_true) == np.array(topic_pred))
    emotion_support = emotion_true_np.sum(axis=0)
    return {
        "topic_accuracy": topic_accuracy,
        "emotion_micro_f1": emotion_micro_f1,
        "emotion_macro_f1": emotion_macro_f1,
        "emotion_per_class_f1": emotion_per_class_f1.tolist(),
        "emotion_support": emotion_support.tolist()
    }