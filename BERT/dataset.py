import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import kagglehub
from transformers import AutoTokenizer
from torch.utils.data import random_split

# For use if necesssary
EMOTION_COLUMNS = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization","relief",
    "remorse","sadness","surprise","neutral"
]
NUM_EMOTIONS = len(EMOTION_COLUMNS)
TOPIC_TYPES = { # NOTE: 0 indexed for compatibility with torch, but 1 indexed in the dataset itself
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
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
    ag_df.loc[:, "topic_label"] = ag_df["topic_label"] - 1 # make the labels 0 indexed so it is compatible with torch processes
    return [go_df, ag_df]


tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

def _tokenize(text, max_length=128):
    """
    Tokenizes the given text based on the requirements of BERT. 
    Returns the tokens in a tensor for processing.
    """
    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

class _MultiTaskDataset(Dataset): # We need this class to manage the properties of the combined dataset that torch will use during training
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        #tokens = _tokenize(row["text"]) # Tokenize the row before passing it back to the trainer

        # Return the item from the dataset, including all the respective fields from the input as well as the tokenizer's added fields
        return {
            "input_ids": row["input_ids"],
            "attention_mask": row["attention_mask"],
            "task": row["task"],
            "emotion_labels": torch.tensor(row["emotion_labels"], dtype=torch.float),
            "topic_label": torch.tensor(row["topic_label"], dtype=torch.long),
        }

def compute_pos_weights(df):
    emotion_rows = df[df["task"] == "emotion"]
    labels = torch.tensor(emotion_rows["emotion_labels"].tolist(), dtype=torch.float)
    
    pos_counts = labels.sum(dim=0)
    total = labels.shape[0]
    neg_counts = total - pos_counts

    # add small epsilon to avoid division by zero
    pos_weight = neg_counts / (pos_counts + 1e-5)
    return pos_weight

# Main Method
def preprocess_data():
    go, ag = _load_datasets()
    combined = pd.concat(_prepare_datafiles(go, ag), ignore_index=True) # combine the datasets
    combined = combined.sample(frac=1).reset_index(drop=True) # shuffles dataset for training process, so we don't accidentally unlearn a task
    pos_weight = compute_pos_weights(combined) # Compute pos weights for emotion task
    tokens = _tokenize(combined["text"].tolist())
    combined["input_ids"] = list(tokens["input_ids"])
    combined["attention_mask"] = list(tokens["attention_mask"])
    dataset = _MultiTaskDataset(combined) 
    train_size = int(0.9 * len(dataset)) #90% training
    val_size = len(dataset) - train_size #10% validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size]) #random split usage
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) #train loader
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False) # val loader
    return train_loader, val_loader, pos_weight



'''
without validation[ORIGINAL]:
def preprocess_data():
    go, ag = _load_datasets()
    combined = pd.concat(_prepare_datafiles(go, ag), ignore_index=True) # combine the datasets
    combined = combined.sample(frac=1).reset_index(drop=True) # shuffles dataset for training process, so we don't accidentally unlearn a task
    dataset = _MultiTaskDataset(combined) 
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    return loader


adapter
def preprocess_data():
    go, ag = _load_datasets()
    go_df, ag_df = _prepare_datafiles(go, ag)
    go_df = go_df.sample(frac=1).reset_index(drop=True) #no longer combine, but keep separate
    ag_df = ag_df.sample(frac=1).reset_index(drop=True)
    emotion_dataset = _MultiTaskDataset(go_df) 
    topic_dataset = _MultiTaskDataset(ag_df)
    emotion_loader = DataLoader(emotion_dataset, batch_size=16, shuffle=True)
    topic_loader = DataLoader(topic_dataset, batch_size=16, shuffle=True)
    return emotion_loader, topic_loader #return two dataloaders to be zipped later
'''