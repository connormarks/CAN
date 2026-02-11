from sklearn.metrics import confusion_matrix, classification_report
from tools.dataset import load_datasets, load_custom_dataset
from tools.preprocess import create_vectorizer, preprocess_go_data, preprocess_ag_data, preprocess_custom_dataset
from tools.train import get_models
from tools.config import MERGED_DATASET_PATH
from collections import namedtuple
import matplotlib.pyplot as plt
import seaborn as sns


Data = namedtuple('Data', ['X_train', 'X_test', 'y_train', 'y_test'])


def process_data(go_data, ag_data, vectorizer):
    print("Preprocessing goemotion data...")
    go_X_train, go_X_test, go_y_train, go_y_test = preprocess_go_data(go_data, vectorizer, fix_class_imbalance=True)
    print("Preprocessing agnews data...")
    ag_X_train, ag_X_test, ag_y_train, ag_y_test = preprocess_ag_data(ag_data, vectorizer)

    GoData = Data(go_X_train, go_X_test, go_y_train, go_y_test)
    AgData = Data(ag_X_train, ag_X_test, ag_y_train, ag_y_test)

    return GoData, AgData


def load_custom_data(vectorizer):
    X, y_emotion, y_topic = load_custom_dataset(MERGED_DATASET_PATH)
    X = preprocess_custom_dataset(X, vectorizer)
    return X, y_emotion, y_topic


def default_scoring(go_model, ag_model, GoData, AgData):
    # Get the confusion matrices
    go_cm = confusion_matrix(GoData.y_test, go_model.predict(GoData.X_test))
    ag_cm = confusion_matrix(AgData.y_test, ag_model.predict(AgData.X_test))

    print("Goemotion classification report:")
    print(classification_report(GoData.y_test, go_model.predict(GoData.X_test)))
    print("Agnews classification report:\n")
    print(classification_report(AgData.y_test, ag_model.predict(AgData.X_test)))

    # Visualize the confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(go_cm, annot=True, fmt='d', cmap='Blues', square=True, linewidths=0.5,
                ax=ax1, cbar_kws={'label': 'Count'}, annot_kws={'size': 6})
    ax1.set_title('GoEmotion Confusion Matrix', fontsize=12)
    ax1.tick_params(axis='both', labelsize=6)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    sns.heatmap(ag_cm, annot=True, fmt='d', cmap='Blues', square=True, linewidths=0.5,
                ax=ax2, cbar_kws={'label': 'Count'})
    ax2.set_title('AG News Confusion Matrix', fontsize=12)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    plt.tight_layout()
    plt.show()


def custom_scoring(go_model, ag_model, X, y_emotion, y_topic):
    go_cm = confusion_matrix(y_emotion, go_model.predict(X))
    ag_cm = confusion_matrix(y_topic, ag_model.predict(X))

    print("Goemotion classification report:")
    print(classification_report(y_emotion, go_model.predict(X)))
    print("Agnews classification report:\n")
    print(classification_report(y_topic, ag_model.predict(X)))

    # Visualize the confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(go_cm, annot=True, fmt='d', cmap='Blues', square=True, linewidths=0.5,
                ax=ax1, cbar_kws={'label': 'Count'}, annot_kws={'size': 6})
    ax1.set_title('GoEmotion Confusion Matrix', fontsize=12)
    ax1.tick_params(axis='both', labelsize=6)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    sns.heatmap(ag_cm, annot=True, fmt='d', cmap='Blues', square=True, linewidths=0.5,
                ax=ax2, cbar_kws={'label': 'Count'})
    ax2.set_title('AG News Confusion Matrix', fontsize=12)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    go_data, ag_data = load_datasets()

    # https://datascience.stackexchange.com/questions/122056/logisticregression-loading-problem
    texts = [*go_data["text"], *ag_data["Description"]]
    vectorizer = create_vectorizer(texts)
    
    GoData, AgData = process_data(go_data, ag_data, vectorizer)

    X, y_emotion, y_topic = load_custom_data(vectorizer)

    go_model, ag_model = get_models(GoData.X_train, GoData.y_train, AgData.X_train, AgData.y_train)

    input("Press Enter to score the default dataset...")
    default_scoring(go_model, ag_model, GoData, AgData)

    input("Press Enter to score the custom dataset...")
    custom_scoring(go_model, ag_model, X, y_emotion, y_topic)
