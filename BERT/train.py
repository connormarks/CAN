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

def train(emotion_loader, topic_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #cuda = GPU, else cpu
    emotion_topic_model = EmotionTopicClassifier().to(device) #attach the device, this is in module.nn's to()
    emotion_topic_model.train() # set the module in training mode, also module.nn

    optimizer = torch.optim.AdamW(emotion_topic_model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    emotion_loss_bce = torch.nn.BCEWithLogitsLoss() #instantiate first outside the loop
    topic_loss_ce = torch.nn.CrossEntropyLoss(ignore_index=config.IGNORE_INDEX)

    for epoch in range(config.NUM_EPOCHS):
        
        for emotion_batch, topic_batch in zip(emotion_loader, topic_loader): #data loading

            optimizer.zero_grad() #clear the gradients every batch

            #EMOTION TRAINING
            emotion_ids = emotion_batch['input_ids'].to(device)
            emotion_mask = emotion_batch['attention_mask'].to(device)
            emotion_labels = emotion_batch['emotion_labels'].to(device)

            logits = emotion_topic_model(emotion_ids, emotion_mask, task='emotion') #forward pass
            emotion_logits = logits['emotion_logits']

            emotion_loss = emotion_loss_bce(emotion_logits,emotion_labels) #task-specific loss

            #TOPIC TRAINING
            topic_ids = topic_batch['input_ids'].to(device)
            topic_mask = topic_batch['attention_mask'].to(device)
            topic_labels = topic_batch['topic_label'].to(device)


            logits = emotion_topic_model(topic_ids, topic_mask, task='topic') #forward pass
            topic_logits = logits['topic_logits']

            topic_loss = topic_loss_ce(topic_logits, topic_labels) #task-specific loss

            #JOINT OPTIMIZATION
            total_loss = emotion_loss + topic_loss #combined loss
            total_loss.backward() #backpropagation            
            optimizer.step() #gradient updates

