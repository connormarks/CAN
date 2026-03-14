# Scoring functions for the few shot baseline
# Authors: Nathan Pietrantonio
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
from config import EMOTION_MAPPING, TOPIC_MAPPING


def _plot_confusion_matrix(cm, title, labels=[], block=False, log_scale=False):
    """
    Plot the confusion matrix

    Inputs:
        cm: numpy array containing the confusion matrix
        title: str containing the title of the plot
        labels: list of labels to use for the plot
        block: whether to block when showing the plot
        log_scale: if True, use logarithmic scale for the color map (zeros shown as 0)
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    if log_scale:
        # LogNorm cannot handle 0; plot cm+1 for colors, annotate with actual counts
        plot_values = cm + 1
        norm = mcolors.LogNorm(vmin=1, vmax=plot_values.max())
        sns.heatmap(plot_values, annot=cm, fmt='d', cmap='Blues', square=True, linewidths=0.5,
                    ax=ax, cbar_kws={'label': 'Count'}, annot_kws={'size': 6},
                    xticklabels=labels, yticklabels=labels, norm=norm)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, linewidths=0.5,
                    ax=ax, cbar_kws={'label': 'Count'}, annot_kws={'size': 6},
                    xticklabels=labels, yticklabels=labels)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.show(block=block)
    plt.pause(0.001)


def custom_scoring(X, y_emotion, y_topic, predicted_emotions, predicted_topics):
    """
    Score the custom dataset and show the data

    Prints the classification report and shows the confusion matrices

    Inputs:
        X: list of text data
        y_emotion: list of true emotion data
        y_topic: list of true topic data
        predicted_emotions: list of predicted emotion data
        predicted_topics: list of predicted topic data
    """
    print("Emotion classification report:")
    print(classification_report(y_emotion, predicted_emotions))
    print("Topic classification report:")
    print(classification_report(y_topic, predicted_topics))

    emotion_labels = list(set(y_emotion))
    topic_labels = list(set(y_topic))

    emotion_cm = confusion_matrix(y_emotion, predicted_emotions, labels=emotion_labels)
    topic_cm = confusion_matrix(y_topic, predicted_topics, labels=topic_labels)

    _plot_confusion_matrix(emotion_cm, 'Emotion Confusion Matrix', emotion_labels, False, True)
    _plot_confusion_matrix(topic_cm, 'Topic Confusion Matrix', topic_labels)


def joint_scoring(X, y_emotion, y_topic, predicted_emotions, predicted_topics):
    """
    Score the joint dataset and show the data

    Prints the classification report and shows the confusion matrices

    Inputs:
        X: list of text data
        y_emotion: list of true emotion data
        y_topic: list of true topic data
        predicted_emotions: list of predicted emotion data
        predicted_topics: list of predicted topic data
    """
    # Build full ordered list of joint labels (same format for data and matrix)
    labels = []
    for emotion in set(y_emotion):
        for topic in set(y_topic):
            labels.append(f"{emotion} about {topic}")

    labels = sorted(labels)

    y_joint = np.array([f"{emotion} about {topic}" for emotion, topic in zip(y_emotion, y_topic)])
    y_joint_pred = np.array([f"{emotion} about {topic}" for emotion, topic in zip(predicted_emotions, predicted_topics)])

    joint_cm = confusion_matrix(y_joint, y_joint_pred, labels=labels)

    print("Joint classification report:")
    print(classification_report(y_joint, y_joint_pred))

    _plot_confusion_matrix(joint_cm, 'Joint Confusion Matrix', labels, False, True)
