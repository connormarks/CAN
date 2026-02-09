import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
import kagglehub

# For use if necesssary
EMOTION_COLUMNS = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization","relief",
    "remorse","sadness","surprise","neutral"
]
NUM_EMOTIONS = len(EMOTION_COLUMNS)
TOPIC_TYPES = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}
NUM_TOPICS = len(TOPIC_TYPES)

def _load_datasets():
    """
    Loads the datasets from the repo and returns their contents
    """
    goemotion_path = kagglehub.dataset_download("debarshichanda/goemotions")
    agnews_path = kagglehub.dataset_download("amananandrai/ag-news-classification-dataset")
    return pd.read_csv(f"{goemotion_path}/data/full_dataset/goemotions_1.csv"), pd.read_csv(f"{agnews_path}/train.csv")

def _prepare_datafiles(go_df, ag_df):
    """
    Takes the given datafiles and edits their columns so they can be merged using the pandas library

    Outputted data files adhere to the following format:
    text: full text associated with the data entry (psot for goemotions, headline + body for agnews)
    emotion_labels: emotion labels if from goemotions, array of -100s if from agnews
    task: what dataset the entry originates from (which head of BERT it will target)
    topic_label: which 
    """
    # NOTE: we use -100 (not None) as a default label so we can know which values to ignore and still use the tensor datastructure later
    go_df = go_df.copy()
    go_df.loc[:, "emotion_labels"] = go_df[EMOTION_COLUMNS].values.tolist()
    go_df = go_df.loc[:, ["text", "emotion_labels"]].copy()
    go_df.loc[:, "task"] = "emotion"
    go_df.loc[:, "topic_label"] = -100 # populate topic label with Null value for goemotions

    ag_df = ag_df.copy()
    ag_df.columns = ["topic_label", "title", "description"]
    ag_df.loc[:, "text"] = ag_df["title"] + " " + ag_df["description"]
    ag_df = ag_df.loc[:, ["text", "topic_label"]].copy()
    ag_df.loc[:, "task"] = "topic"
    ag_df.loc[:, "emotion_labels"] = [[-100] * NUM_EMOTIONS for _ in range(len(ag_df))] # populate emotion labels with Null for ag news
    return [go_df, ag_df]

# Main Method
def preprocess_data():
    go, ag = _load_datasets()
    combined = pd.concat(_prepare_datafiles(go, ag), ignore_index=True) # combine the datasets
    combined = combined.sample(frac=1).reset_index(drop=True) # shuffles dataset for training process, so we don't accidentally unlearn a task
    return combined

# For testing, remove when done
print(preprocess_data())