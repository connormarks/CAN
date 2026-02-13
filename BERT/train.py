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

    optimizer = torch.optim.AdamW(emotion_topic_model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    emotion_loss_bce = torch.nn.BCEWithLogitsLoss() #instantiate first outside the loop
    topic_loss_ce = torch.nn.CrossEntropyLoss(ignore_index=config.IGNORE_INDEX)

    for epoch in range(config.NUM_EPOCHS):
        
        for batch in dataloader: #data loading
            optimizer.zero_grad() #clear the gradients every batch

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            emotion_labels = batch['emotion_labels'].to(device)
            topic_labels = batch['topic_label'].to(device)


            logits = emotion_topic_model(input_ids, attention_mask) #forward pass
            emotion_logits = logits['emotion_logits']
            topic_logits = logits['topic_logits']

            emotion_mask = (emotion_labels != -100).any(dim=1)
            emotion_loss = emotion_loss_bce(emotion_logits[emotion_mask], emotion_labels[emotion_mask]) if emotion_mask.any() else torch.tensor(0.0, device=device) #task-specific loss
            topic_loss = topic_loss_ce(topic_logits, topic_labels) #task-specific loss

            total_loss = emotion_loss + topic_loss # scalar tensor with one value inside (combine loss)

            total_loss.backward() #backpropagation

            torch.nn.utils.clip_grad_norm_(emotion_topic_model.parameters(), max_norm=1.0) # gradient clipping

            optimizer.step() #gradient updates

