import json
from pathlib import Path


PROCESSED_PATH = Path(
    "data/processed/spiqa_testA_processed.json"
)


with open(PROCESSED_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


total = len(data)
found = 0
missing = 0


for item in data:

    image_path = Path(item["image_path"])

    if image_path.exists():
        found += 1
    else:
        missing += 1
        print("Missing image:", image_path)


print("\nImage verification complete!")
print("Total QA pairs:", total)
print("Images found:", found)
print("Images missing:", missing)