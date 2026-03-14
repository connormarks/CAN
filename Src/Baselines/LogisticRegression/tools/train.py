# Authors: Nathan Pietrantonio
from sklearn.linear_model import LogisticRegression
from .config import MODEL_PATH
import pickle

def train_model(X_train, y_train, max_iter=100, class_weight=None, save_name=None):
    """
    Trains the Logistic Regression model on the data

    Inputs:
        X_train: list - The training feature data
        y_train: list - The training target data
        max_iter: int - The maximum number of iterations to run
        class_weight: dict - The class weights to use
        save_name: str - The name to save the model as

    Returns:
        model: LogisticRegression model - The trained model
        If save_name is provided, the model is saved to the MODEL_PATH folder
    """
    model = LogisticRegression(max_iter=max_iter, class_weight=class_weight)
    model.fit(X_train, y_train)
    if save_name:
        with open(f'{MODEL_PATH}/{save_name}.pkl', 'wb') as f:
            pickle.dump(model, f)
    return model


def get_models(go_X_train, go_y_train, ag_X_train, ag_y_train):
    """
    Gets the goemotion and agnews models, either loading from a file or training them

    Inputs:
        go_X_train: list - The training feature data for the goemotion model
        go_y_train: list - The training target data for the goemotion model
        ag_X_train: list - The training feature data for the agnews model
        ag_y_train: list - The training target data for the agnews model

    Returns:
        go_model: LogisticRegression model - The goemotion model
        ag_model: LogisticRegression model - The agnews model
    """
    load_go_model = input("Load goemotion model? (y/n): ")
    load_ag_model = input("Load agnews model? (y/n): ")
    print()
    if load_go_model == "y":
        go_model = pickle.load(open(f'{MODEL_PATH}/go_model.pkl', "rb"))
    else:
        print("Fitting goemotion model...")
        go_model = train_model(go_X_train, go_y_train, max_iter=200, class_weight="balanced", save_name="go_model")
    if load_ag_model == "y":
        ag_model = pickle.load(open(f'{MODEL_PATH}/ag_model.pkl', "rb"))
    else:
        print("Fitting agnews model...\n")
        ag_model = train_model(ag_X_train, ag_y_train, max_iter=200, class_weight="balanced", save_name="ag_model")

    return go_model, ag_model
