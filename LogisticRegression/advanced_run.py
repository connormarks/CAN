from sklearn.metrics import confusion_matrix, classification_report
from tools.dataset import load_datasets, create_test_train_split
from tools.preprocess import preprocess_go_data, preprocess_ag_data
from tools.train import get_models
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    go_data, ag_data = load_datasets()

    print("Preprocessing goemotion data...")
    go_X_train, go_X_test, go_y_train, go_y_test = preprocess_go_data(go_data, fix_class_imbalance=True)
    print("Preprocessing agnews data...")
    ag_X_train, ag_X_test, ag_y_train, ag_y_test = preprocess_ag_data(ag_data)

    return go_X_train, go_X_test, go_y_train, go_y_test, ag_X_train, ag_X_test, ag_y_train, ag_y_test


def score_models(go_model, ag_model, go_X_test, go_y_test, ag_X_test, ag_y_test):
    # Get the confusion matrices
    go_cm = confusion_matrix(go_y_test, go_model.predict(go_X_test))
    ag_cm = confusion_matrix(ag_y_test, ag_model.predict(ag_X_test))

    print("Goemotion classification report:")
    print(classification_report(go_y_test, go_model.predict(go_X_test)))
    print("Agnews classification report:\n")
    print(classification_report(ag_y_test, ag_model.predict(ag_X_test)))

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
    # There has to be a better way to do this...
    go_X_train, go_X_test, go_y_train, go_y_test, ag_X_train, ag_X_test, ag_y_train, ag_y_test = load_data()

    go_model, ag_model = get_models(go_X_train, go_y_train, ag_X_train, ag_y_train)

    score_models(go_model, ag_model, go_X_test, go_y_test, ag_X_test, ag_y_test)
