# Authors: Nathan Pietrantonio
from sklearn.metrics import accuracy_score
from tools.dataset import load_datasets, create_test_train_split
from tools.preprocess import preprocess_go_data, preprocess_ag_data, create_vectorizer
from tools.train import train_model

if __name__ == "__main__":
    # Quickly runs the models with very little preprocessing and analysis
    # For more detailed analysis, see advanced_run.py

    print("FOR MORE DETAILED ANALYSIS, SEE advanced_run.py\n")

    go_data, ag_data = load_datasets()

    texts = [*go_data["text"], *ag_data["Description"]]
    vectorizer = create_vectorizer(texts)

    print("Preprocessing goemotion data...")
    go_X_train, go_X_test, go_y_train, go_y_test = preprocess_go_data(go_data, vectorizer)
    print("Preprocessing agnews data...")
    ag_X_train, ag_X_test, ag_y_train, ag_y_test = preprocess_ag_data(ag_data, vectorizer)

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
