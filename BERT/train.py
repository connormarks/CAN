#traning loop

import config
from model import EmotionTopicClassifier
from dataset import preprocess_data
import numpy as np
import torch



def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    emotion_topic_model = EmotionTopicClassifier().to(device) 
    emotion_topic_model.train() # set the module in training mode