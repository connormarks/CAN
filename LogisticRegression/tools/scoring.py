from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tools.config import EMOTION_MAPPING, TOPIC_MAPPING


def _plot_confusion_matrix(cm, title, labels=[]):
    """
    Plot the confusion matrix

    Inputs:
        cm: numpy array containing the confusion matrix
        title: str containing the title of the plot
        labels: list of labels to use for the plot
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, linewidths=0.5,
                ax=ax, cbar_kws={'label': 'Count'}, annot_kws={'size': 6}, xticklabels=labels, yticklabels=labels)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)


def default_scoring(go_model, ag_model, GoData, AgData):
    """
    Score the default dataset and show the data

    Prints the classification report and shows the confusion matrices

    Inputs:
        go_model: LogisticRegression model for the goemotion data
        ag_model: LogisticRegression model for the agnews data
        GoData: Data namedtuple containing the training and test data for the goemotion model
        AgData: Data namedtuple containing the training and test data for the agnews model
    """
    print("Goemotion classification report:")
    print(classification_report(GoData.y_test, go_model.predict(GoData.X_test)))
    print("Agnews classification report:\n")
    print(classification_report(AgData.y_test, ag_model.predict(AgData.X_test)))

    go_cm = confusion_matrix(GoData.y_test, go_model.predict(GoData.X_test))
    ag_cm = confusion_matrix(AgData.y_test, ag_model.predict(AgData.X_test))

    emotion_labels = [list(EMOTION_MAPPING.keys())[emotion] for emotion in go_model.classes_]
    topic_labels = [list(TOPIC_MAPPING.keys())[topic-1] for topic in ag_model.classes_]

    _plot_confusion_matrix(go_cm, 'GoEmotion Confusion Matrix', emotion_labels)
    _plot_confusion_matrix(ag_cm, 'AG News Confusion Matrix', topic_labels)


def custom_scoring(go_model, ag_model, X, y_emotion, y_topic):
    """
    Score the custom dataset and show the data

    Prints the classification report and shows the confusion matrices

    Inputs:
        go_model: LogisticRegression model for the goemotion data
        ag_model: LogisticRegression model for the agnews data
        X: numpy array containing the preprocessed text data
        y_emotion: numpy array containing the preprocessed emotion data
        y_topic: numpy array containing the preprocessed topic data
    """
    print("Goemotion classification report:")
    print(classification_report(y_emotion, go_model.predict(X)))
    print("Agnews classification report:\n")
    print(classification_report(y_topic, ag_model.predict(X)))

    go_cm = confusion_matrix(y_emotion, go_model.predict(X))
    ag_cm = confusion_matrix(y_topic, ag_model.predict(X))

    emotion_labels = [list(EMOTION_MAPPING.keys())[emotion] for emotion in go_model.classes_]
    topic_labels = [list(TOPIC_MAPPING.keys())[topic-1] for topic in ag_model.classes_]

    _plot_confusion_matrix(go_cm, 'GoEmotion Confusion Matrix', emotion_labels)
    _plot_confusion_matrix(ag_cm, 'AG News Confusion Matrix', topic_labels)


def joint_scoring(go_model, ag_model, X, y_emotion, y_topic):
    """
    Score the joint dataset and show the data

    Prints the classification report and shows the confusion matrices

    Inputs:
        go_model: LogisticRegression model for the goemotion data
        ag_model: LogisticRegression model for the agnews data
        X: numpy array containing the preprocessed text data
        y_emotion: numpy array containing the preprocessed emotion data
        y_topic: numpy array containing the preprocessed topic data
    """
    # Build full ordered list of joint labels (same format for data and matrix)
    labels = []
    for emotion in go_model.classes_:
        for topic in ag_model.classes_:
            emotion_name = list(EMOTION_MAPPING.keys())[emotion]
            topic_name = list(TOPIC_MAPPING.keys())[topic - 1]
            labels.append(f"{emotion_name} about {topic_name}")

    def to_joint_label(emotion_idx, topic_idx):
        en = list(EMOTION_MAPPING.keys())[emotion_idx]
        tn = list(TOPIC_MAPPING.keys())[topic_idx - 1]
        return f"{en} about {tn}"

    y_joint = np.array([to_joint_label(e, t) for e, t in zip(y_emotion, y_topic)])
    y_joint_pred = np.array([to_joint_label(e, t) for e, t in zip(go_model.predict(X), ag_model.predict(X))])

    joint_cm = confusion_matrix(y_joint, y_joint_pred, labels=labels)

    print("Joint classification report:")
    print(classification_report(y_joint, y_joint_pred))

    _plot_confusion_matrix(joint_cm, 'Joint Confusion Matrix', labels)
