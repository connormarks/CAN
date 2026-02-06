import os
import json

OUTPUT_DIR = 'Output'
VERIFIED_DIR = f'{OUTPUT_DIR}/Processed'

def select_output_file():
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
    for obj in json_obj:
        print(obj['text'])
        print(obj['emotion'])
        print(obj['topic'])
        print()
        verified = input("Is this data verified? (Enter for yes, 'n' for no): ")
        print()
        if verified == '':
            obj['verified'] = True
    return json_obj

if __name__ == "__main__":
    output_file = select_output_file()
    json_obj = json.load(open(f'{OUTPUT_DIR}/{output_file}'))
    verified_json_obj = verify_data(json_obj)
    with open(f'{VERIFIED_DIR}/{output_file}', 'w') as file:
        json.dump(verified_json_obj, file, indent=4)
    print(f"Data verified and saved to {VERIFIED_DIR}/{output_file}")
