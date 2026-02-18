# BERT base model + emotion and topic heads

import torch.nn as nn
from transformers import BertModel # huggingface

# our bert model
class EmotionTopicClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("google-bert/bert-base-uncased") # THIS IS THE MODEL FROM TRANSFORMERS.

        #self.bert.add_adapter('emotion') # task-specific adapters for training
        #self.bert.add_adapter('topic')

        #self.bert.train_adapter(['emotion', 'topic']) #this line freezes the base model (making training more efficient)

        hidden = self.bert.config.hidden_size # tracks size of hidden layer vectors for training

        # Create the two heads for training; one for emotions, one for topic
        self.emotion_head = nn.Linear(hidden, 28) 
        self.topic_head = nn.Linear(hidden, 4)


    def forward(self, input_ids, attention_mask): # used by torch in the training process

        #self.bert.set_active_adapters(task)

        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = out.last_hidden_state[:, 0] # gets the first token for all samples in the batch

        return {
            "emotion_logits": self.emotion_head(cls),
            "topic_logits": self.topic_head(cls)
        }