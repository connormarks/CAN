from sklearn.metrics import accuracy_score
from tools.dataset import load_datasets, create_test_train_split
from tools.preprocess import preprocess_go_data, preprocess_ag_data
from tools.train import train_model

if __name__ == "__main__":
    go_data, ag_data = load_datasets()

    print("Preprocessing goemotion data...")
    go_X, go_y = preprocess_go_data(go_data)
    print("Preprocessing agnews data...\n")
    ag_X, ag_y = preprocess_ag_data(ag_data)

    print("Creating test train split...\n")
    go_X_train, go_X_test, go_y_train, go_y_test = create_test_train_split(go_X, go_y)
    ag_X_train, ag_X_test, ag_y_train, ag_y_test = create_test_train_split(ag_X, ag_y)

    print("Fitting goemotion model...")
    go_model = train_model(go_X_train, go_y_train, max_iter=200)
    print("Fitting agnews model...\n")
    ag_model = train_model(ag_X_train, ag_y_train)

    print("Scoring goemotion model...")
    go_score = accuracy_score(go_y_test, go_model.predict(go_X_test))
    print("Scoring agnews model...\n")
    ag_score = accuracy_score(ag_y_test, ag_model.predict(ag_X_test))

    print(f"Goemotion accuracy: {go_score}")
    print(f"Agnews accuracy: {ag_score}")
