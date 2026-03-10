TOPIC = {
    "World": 0,
    "Sports": 1,
    "Business": 2,
    "Sci/Tech": 3,
    }

TOPIC_ID = {v: k for k, v in TOPIC.items()}

EKMAN_MAPPING = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": [
        "joy", "amusement", "approval", "excitement", "gratitude",
        "love", "optimism", "relief", "pride", "admiration",
        "desire", "caring"
    ],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"],
    }

EMOTION_NAMES = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise"
    ]

EMOTION_ID = {name: i for i, name in enumerate(EMOTION_NAMES)}

EKMAN_DICT= {}
for new, old in EKMAN_MAPPING.items():
    for emo in old:
        EKMAN_DICT[emo] = new

