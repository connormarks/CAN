#BERT base model + emotion and topic heads

import torch
import torch.nn as nn
from transformers import BertModel # huggingface


#our bert model
class EmotionTopicClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("google-bert/bert-base-uncased") #THIS IS THE MODEL FROM TRANSFORMERS.

        self.emotion_head = nn.Linear(768, 28)
        self.topic_head = nn.Linear(768, 4)






if __name__ == "__main__":
    our_bert = EmotionTopicClassifier()

    print('\nEmotion Topic Classifier embeddings:\n')
    print(our_bert)

    print('\nPre-trained model (baseline):\n')
    print(our_bert.bert)
    
    print('\nconfig:\n')
    print(our_bert.bert.config)
