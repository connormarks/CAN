import os
import json
from config import OUTPUT_DIR, VERIFIED_DIR, EMOTION_LABELS, TOPIC_LABELS


def _output_information(obj):
    """

    """
    print(" | Text:")
    print(f" | \t{obj['text']}")
    print(" | Emotion:")
    print(f" | \t{obj['emotion']}")
    print(" | Topic:")
    print(f" | \t{obj['topic']}")
    print()


def _verify_emotion(obj):
    """
    Verify the emotion in the emotion list.
    Returns the verified emotion list.

    Input:
        emotion: list - The emotion list to verify.
    Returns:
        emotion: list - The verified emotion list.
    """
    for emotion in obj['emotion']:
        if emotion not in EMOTION_LABELS:
            return False
    return True


def _verify_topic(obj):
    """
    Verify the topic in the topic list.
    Returns the verified topic list.
    """
    if obj['topic'] not in TOPIC_LABELS:
        return False
    return True


def _verify_object(obj):
    if not _verify_emotion(obj):
        print(f"Skipping, invalid emotion\n")
        return None
    if not _verify_topic(obj):
        print(f"Skipping, invalid topic\n")
        return None

    verified = input("Is this data verified? (Enter for yes, 'n' for no, 'e' for needs editing): ")
    if verified == '':
        obj['needs_editing'] = False
        obj['verified'] = True
    elif verified == 'e':
        obj['needs_editing'] = True
        obj['verified'] = True
    else:
        obj['needs_editing'] = None
        obj['verified'] = False
    print()
    return obj


def select_output_file():
    """
    Prompt the user to select an output file from the available files in the output directory.
    Returns the selected file name.

    Returns:
        output_file: str - The selected file name.
    """
    output_files = [file for file in os.listdir(OUTPUT_DIR) if file.endswith('.json')]
    print()
    print("Available output files:")
    for index, file in enumerate(output_files):
        print(f'\t{index}. {file}')
    print()
    choice = input("Enter the number of the file you want to verify: ")
    try:
        choice = int(choice)
        output_file = output_files[int(choice)]
    except Exception:
        print(f"Invalid choice")
        return select_output_file()
    print(f"Selected file: {output_file}")
    print()
    return output_file


def verify_data(json_obj):
    """
    Verify the data in the JSON object.
    Returns the verified JSON object.

    Input:
        json_obj: list - The JSON object to verify.
    Returns:
        json_obj: list - The verified JSON object.
    """
    print(f"Verifying {len(json_obj)} examples")
    print()
    verified_json_objs = []
    for obj in json_obj:
        _output_information(obj)
        obj = _verify_object(obj)
        if obj is None:
            continue
        verified_json_objs.append(obj)
    return verified_json_objs


if __name__ == "__main__":
    output_file = select_output_file()
    json_obj = json.load(open(f'{OUTPUT_DIR}/{output_file}'))
    verified_json_obj = verify_data(json_obj)
    with open(f'{VERIFIED_DIR}/{output_file}', 'w') as file:
        json.dump(verified_json_obj, file, indent=4)
    print(f"Data verified and saved to {VERIFIED_DIR}/{output_file}")
