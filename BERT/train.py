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
    task-specific loss
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

