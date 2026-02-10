#BERT base model + emotion and topic heads

import torch
import torch.nn as nn
from transformers import BertModel # huggingface


#our bert model
class EmotionTopicClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        #base model
        self.bert = BertModel.from_pretrained("google-bert/bert-base-uncased") #768 hidden layers (check the config)

        #heads
        self.emotion_head = nn.Linear(768, 28) # Emotion Neural Net Head (28 outputs)
        self.topic_head = nn.Linear(768, 4) # Topic Neural Net Head (4 outputs)


    def forward(self, input_ids, attention_mask):
        '''
            Given tokenized text, produces emotion logits and topic logits.
        '''
        contextual_embeddings = self.bert(input_ids = input_ids, attention_mask = attention_mask) #run through bert, produces contextual embeddings
        cls_representation = contextual_embeddings.last_hidden_state[:, 0]

        emotion_logits = self.emotion_head(cls_representation)
        topic_logits = self.topic_head(cls_representation)

        return emotion_logits, topic_logits #




if __name__ == "__main__":
    our_bert = EmotionTopicClassifier()

    print('\nEmotion Topic Classifier embeddings:\n')
    print(our_bert)

    print('\nPre-trained model (baseline):\n')
    print(our_bert.bert)
    
    print('\nconfig:\n')
    print(our_bert.bert.config)

    print('\nBert Signature:\n')
    print(help(BertModel.forward))
