#hyperparameters, label counts

MODEL_NAME = 'google-bert/bert-base-uncased' #hugging face

MAX_SEQUENCE_LENGTH = 128 #max number of tokens per seq, longer is truncated.
BATCH_SIZE = 16 #num of examples processed together in one forward pass
SHUFFLE = True #shuffle to make learning more pure

LEARNING_RATE = 2e-5 #learning rate for the optimizer FROM THE BERT PAPER
WEIGHT_DECAY = 0.01 #weight decay coeff for adamW, L2 reg
NUM_EPOCHS = 3 # num full passes over training data

WARMUP_RATIO = 0.1 #fraction of total traning steps for learning rate

EMOTION_LOSS_TYPE = 'bce' #BCEWithLogitsLoss, multi label
TOPIC_LOSS_TYPE = 'cross_entropy' #CrossEntropyLoss, single label
IGNORE_INDEX = -100 #Missing Labels have no effect

EMOTION_LOSS_WEIGHT = 1.0 #Increasing this value makes the model prioritize emotion prediction more
TOPIC_LOSS_WEIGHT = 1.0 #Increasing this value makes the model prioritize topic prediction more

RANDOM_SEED = 42 #for reproducibility
USE_CUDA = True #GPU
LOG_N_STEPS = 50 #num training steps between logging the update (slows training)