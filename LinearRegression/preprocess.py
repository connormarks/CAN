from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(text, max_df=0.95, min_df=2, max_features=10000, stop_words='english'):
    """
    Vectorizes the text using the TfidfVectorizer
    """
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=max_features, stop_words=stop_words)
    return vectorizer.fit_transform(text)


def _map_go_emotion_to_index(emotion_columns):
    print(emotion_columns)
    exit()


def preprocess_go_data(go_data):
    # Filter out unclear examples
    go_data = go_data[go_data["example_very_unclear"] == False]

    columns_to_ignore = [ 
        "id", 
        "author", 
        "subreddit", 
        "link_id", 
        "parent_id", 
        "created_utc", 
        "rater_id", 
        "example_very_unclear"
    ]

    # Create X and y
    texts = go_data["text"]
    X = vectorize_text(texts)
    y = go_data.drop(columns=["text"] + columns_to_ignore)
    y = _map_go_emotion_to_index(y)
    print(y)
    exit()
    return X, y


def preprocess_ag_data(ag_data):
    """
    Preprocesses the agnews dataset
    """
    descriptions = ag_data["Description"]
    X = vectorize_text(descriptions)
    y = ag_data[["Class Index"]]
    return X, y
