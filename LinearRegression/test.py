from sklearn.linear_model import LinearRegression
from dataset import load_datasets, create_test_train_split
from preprocess import preprocess_go_data, preprocess_ag_data

def fit_model(data, target):
    """
    Fits the Linear Regression model to the data
    """
    model = LinearRegression()
    model.fit(data, target)
    return model


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
    go_model = fit_model(go_X_train, go_y_train)
    print("Fitting agnews model...\n")
    ag_model = fit_model(ag_X_train, ag_y_train)

    print("Scoring goemotion model...")
    go_score = go_model.score(go_X_test, go_y_test)
    print("Scoring agnews model...\n")
    ag_score = ag_model.score(ag_X_test, ag_y_test)

    print(f"Goemotion score: {go_score}")
    print(f"Agnews score: {ag_score}")
