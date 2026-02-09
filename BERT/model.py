#BERT base model + emotion and topic heads

import torch
import torch.nn as nn
from transformers import BertModel # huggingface


#our bert model
class EmotionTopicClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("google-bert/bert-base-uncased") #THIS IS THE MODEL FROM TRANSFORMERS.





if __name__ == "__main__":
    our_bert = EmotionTopicClassifier()
    print(our_bert)