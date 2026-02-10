from sklearn.linear_model import LogisticRegression
import pickle

MODEL_PATH = "models"

def train_model(X_train, y_train, max_iter=100, save_name=None):
    """
    Trains the Logistic Regression model on the data
    """
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    if save_name:
        with open(f'{MODEL_PATH}/{save_name}.pkl', 'wb') as f:
            pickle.dump(model, f)
    return model


def get_models(go_X_train, go_y_train, ag_X_train, ag_y_train):
    load_go_model = input("Load goemotion model? (y/n): ")
    load_ag_model = input("Load agnews model? (y/n): ")
    print()
    if load_go_model == "y":
        go_model = pickle.load(open(f'{MODEL_PATH}/go_model.pkl', "rb"))
    else:
        print("Fitting goemotion model...")
        go_model = train_model(go_X_train, go_y_train, max_iter=200, save_name="go_model")
    if load_ag_model == "y":
        ag_model = pickle.load(open(f'{MODEL_PATH}/ag_model.pkl', "rb"))
    else:
        print("Fitting agnews model...\n")
        ag_model = train_model(ag_X_train, ag_y_train, save_name="ag_model")

    return go_model, ag_model
