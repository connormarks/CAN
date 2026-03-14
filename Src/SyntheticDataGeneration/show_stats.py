# Authors: Nathan Pietrantonio
from config import MERGED_DIR
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import json


EMOTION_COUNTS_FILE = f'{MERGED_DIR}/emotion_counts.json'
TOPIC_COUNTS_FILE = f'{MERGED_DIR}/topic_counts.json'
JOINT_COUNTS_FILE = f'{MERGED_DIR}/joint_counts.json'


def get_class_counts(json_obj):
    print(f"Getting class counts for {len(json_obj)} examples...")
    emotion_counts = Counter()
    topic_counts = Counter()
    joint_counts = Counter()
    for obj in json_obj:
        if not obj['verified']:
            continue

        primary_emotion = obj['emotion'][0]
        emotion_counts[primary_emotion] += 1
        topic_counts[obj['topic']] += 1
        joint_counts[f"{primary_emotion} about {obj['topic']}"] += 1

    return emotion_counts, topic_counts, joint_counts


def plot_class_counts(counter, title):
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(counter.keys()), y=list(counter.values()))
    plt.title(title, fontsize=16)
    plt.xlabel('Class', fontsize=13)
    plt.ylabel('Count', fontsize=13)
    plt.xticks(rotation=45, ha='right')
    plt.tick_params(axis='x', labelsize=11)
    plt.tick_params(axis='y', labelsize=11)
    plt.tight_layout()
    plt.show(block=False)


def save_report(emotion_counts, topic_counts, joint_counts):
    sorted_emotions = {}
    sorted_topics = {}
    sorted_joints = {}
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        sorted_emotions[emotion] = count
    for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
        sorted_topics[topic] = count
    for joint, count in sorted(joint_counts.items(), key=lambda x: x[1], reverse=True):
        sorted_joints[joint] = count

    with open(EMOTION_COUNTS_FILE, 'w') as f:
        json.dump(sorted_emotions, f, indent=4)
    with open(TOPIC_COUNTS_FILE, 'w') as f:
        json.dump(sorted_topics, f, indent=4)
    with open(JOINT_COUNTS_FILE, 'w') as f:
        json.dump(sorted_joints, f, indent=4)

if __name__ == "__main__":
    with open(f'{MERGED_DIR}/merged_data.json', 'r') as f:
        json_obj = json.load(f)
    emotion_counts, topic_counts, joint_counts = get_class_counts(json_obj)
    plot_class_counts(emotion_counts, 'Synthetic Dataset Emotion Counts')
    plot_class_counts(topic_counts, 'Synthetic Dataset Topic Counts')
    plot_class_counts(joint_counts, 'Synthetic Dataset Joint Counts')
    input()
    save_report(emotion_counts, topic_counts, joint_counts)
    print(f"Report saved to {MERGED_DIR}/emotion_counts.json, {MERGED_DIR}/topic_counts.json, {MERGED_DIR}/joint_counts.json")
