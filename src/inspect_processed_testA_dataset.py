import os
import json


# Path to processed dataset
DATA_PATH = "data/processed"


def inspect_file(file_path):
    print("\n" + "=" * 60)
    print(f"FILE: {file_path}")
    print("=" * 60)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print("Type:", type(data))

        if isinstance(data, list):
            print("Number of samples:", len(data))

            if len(data) > 0:
                print("\nKeys in first sample:")
                print(data[0].keys())

                print("\nFirst sample:")
                print(json.dumps(data[0], indent=2, ensure_ascii=False))

        elif isinstance(data, dict):
            print("Dictionary keys:")
            print(data.keys())

            print("\nData:")
            print(json.dumps(data, indent=2, ensure_ascii=False))

    except Exception as e:
        print("Error:", e)


def main():

    print("=" * 60)
    print("SPIQA PROCESSED DATASET INSPECTION")
    print("=" * 60)

    print("\nLooking inside:", DATA_PATH)

    if not os.path.exists(DATA_PATH):
        print("\nERROR: Processed data folder not found!")
        return

    print("\nFiles/folders found:")

    for item in os.listdir(DATA_PATH):
        print(" -", item)

    # Inspect JSON files
    for root, dirs, files in os.walk(DATA_PATH):

        for file in files:

            if file.endswith(".json"):

                file_path = os.path.join(root, file)

                inspect_file(file_path)


if __name__ == "__main__":
    main()