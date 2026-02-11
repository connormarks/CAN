#traning loop

import config
from model import EmotionTopicClassifier
from dataset import preprocess_data
import numpy as np
import torch

'''
-----Training loop notes----

essential:
    data loading
    forward pass
    task-specific loss (then combine)
    backpropagate
    optimizer

advanced techniques:
    task-specific adapters
    gradient projection
    dynamic loss-reweighting
    task-specific schedulers
'''

def train(dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #cuda = GPU, else cpu
    emotion_topic_model = EmotionTopicClassifier().to(device) #attach the device, this is in module.nn's to()
    emotion_topic_model.train() # set the module in training mode, also module.nn

    for epoch in config.NUM_EPOCHS:
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            emotion_labels = batch['emotion_labels'].to(device)
            topic_labels = batch['topic_label'].to(device)


            logits = emotion_topic_model.forward(input_ids, attention_mask) #logits is a dictionary
            emotion_logits = logits['emotion_logits']
            topic_logits = logits['topic_logits']

            emotion_loss = torch.nn.BCEWithLogitsLoss(emotion_logits,emotion_labels)
            topic_loss = torch.nn.CrossEntropyLoss(topic_logits, topic_labels, ignore_index=config.IGNORE_INDEX)

            total_loss = emotion_loss + topic_loss


  
