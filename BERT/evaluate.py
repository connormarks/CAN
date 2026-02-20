import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def evaluate(model, loader, device) -> None:
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

            emotion_probs = torch.sigmoid(emotion_logits)
            emotion_preds = (emotion_probs > 0.5).int()

            # assembles the cm array with both predictions encapsulated
            for i in range(len(topic_labels)):
                true_topic = topic_labels[i].item()
                pred_topic = topic_preds[i].item()

                true_emotion = ''.join(
                    map(str, emotion_labels[i].int().cpu().numpy())
                )
                pred_emotion = ''.join(
                    map(str, emotion_preds[i].cpu().numpy())
                )

                joint_true.append(f"{true_topic}_{true_emotion}")
                joint_pred.append(f"{pred_topic}_{pred_emotion}")

    labels = sorted(list(set(joint_true + joint_pred))) # gets all labels based on validation set; this wasn't defined in LR cm functions
    cm = confusion_matrix(joint_true, joint_pred, labels=labels)
    print("Done computing validation set! outputting chart...")
    
    # assembles and outputs graph of CM heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
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
    plt.title("Joint Topic + Emotion Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    model.train()