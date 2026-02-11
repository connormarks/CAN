# BERT base model + emotion and topic heads

import torch
import torch.nn as nn
from transformers import BertModel # huggingface
from torch.utils.data import DataLoader
from dataset import preprocess_data

# our bert model
class EmotionTopicClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("google-bert/bert-base-uncased") # THIS IS THE MODEL FROM TRANSFORMERS.

        self.bert.add_adapter('emotion') # task-specific adapters for training
        self.bert.add_adapter('topic')

        self.bert.train_adapter(['emotion', 'topic']) #this line freezes the base model (making training more efficient)

        hidden = self.bert.config.hidden_size # tracks size of hidden layer vectors for training

        # Create the two heads for training; one for emotions, one for topic
        self.emotion_head = nn.Linear(hidden, 28) 
        self.topic_head = nn.Linear(hidden, 4)


    def forward(self, input_ids, attention_mask, task): # used by torch in the training process

        self.bert.set_active_adapters(task)

        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = out.last_hidden_state[:, 0] # gets the first token for all samples in the batch

        return {
            "emotion_logits": self.emotion_head(cls),
            "topic_logits": self.topic_head(cls)
        }









# NOTE: Don't use this in final, I generated it with ChatGPT. I just used this to validate the preprocessing methods.
# leaving it here for now for testing purposes
def train(training_data: DataLoader, num_epochs=1):
    # Initialize model, optimizer, loss
    model = EmotionTopicClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    emotion_loss_fn = torch.nn.BCEWithLogitsLoss()
    topic_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100) # using the aforementioned -100 as a Null value

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch in training_data:
            # Move tensors to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            emotion_labels = batch["emotion_labels"].to(device)
            topic_labels = batch["topic_label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)

            # Pick the correct head
            if batch["task"][0] == "emotion":
                loss = emotion_loss_fn(outputs["emotion_logits"], emotion_labels)
            else:
                loss = topic_loss_fn(outputs["topic_logits"], topic_labels)

            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1} done")

# for testing purposes, remove when done
train(preprocess_data())
