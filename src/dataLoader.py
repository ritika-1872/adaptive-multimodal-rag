import json
from pathlib import Path


class SPIQADataset:

    def __init__(self, json_path):
        self.json_path = Path(json_path)

        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def keys(self):
        """Return the top-level keys in the dataset."""
        return self.data.keys()

    def get(self, key):
        """Return the value associated with a top-level key."""
        return self.data[key]


if __name__ == "__main__":

    dataset = SPIQADataset(
        "data/test-A/SPIQA_testA.json"
    )

    print("Dataset loaded successfully!")

    print("\nData type:")
    print(type(dataset.data))

    print("\nTop-level keys:")
    for key in dataset.keys():
        print(" -", key)

    print("\nNumber of top-level entries:")
    print(len(dataset.data))

    # Get the first top-level entry
    first_key = next(iter(dataset.data))

    print("\nFirst top-level key:")
    print(first_key)

    print("\nContent of first entry:")
    print(dataset.data[first_key])