import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# used in matrix charting
TOPIC_MAPPING = {
    "World": 1,
    "Sports": 2,
    "Business": 3,
    "Sci/Tech": 4,
}
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
}

def evaluate(model, loader, device, run_dir, epoch) -> None:
    """
    Calculates and outputs the confusion matrix on the validation data.
    Confusion Matrix is calculated with sklearn and output with seaborn
    """
    model.eval()

    joint_true = []
    joint_pred = []

    with torch.no_grad():
        # copied from validation loop in train.py and modified for cm computation
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

            # assembles the cm array with both predictions encapsulated
            for i in range(len(topic_labels)):
                true_topic = topic_labels[i].item()
                pred_topic = topic_preds[i].item()
                true_emotion = topic_labels[i].item()
                pred_emotion = topic_preds[i].item()

                joint_true.append(f"{true_topic} about {true_emotion}")
                joint_pred.append(f"{pred_topic} about {pred_emotion}")
    
    labels = []
    for emotion in EMOTION_MAPPING.keys():
        for topic in TOPIC_MAPPING.keys():
            labels.append(f"{emotion} about {topic}")

    joint_cm = confusion_matrix(joint_true, joint_pred, labels=labels)

    # assembles and outputs graph of CM heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        joint_cm,
        cmap="Blues",
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Count"},
        annot_kws={"size": 6}
    )
    plt.title(f"Topic + Emotion Confusion Matrix: Epoch {epoch}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_path = os.path.join(run_dir, f"confusion_matrix{epoch}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    model.train()