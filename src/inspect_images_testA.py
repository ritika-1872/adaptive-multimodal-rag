import json
import os
import random


CORPUS_PATH = "data/processed/evidence_corpus.json"


def main():

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    print("=" * 60)
    print("VISUAL EVIDENCE INSPECTION")
    print("=" * 60)

    print(f"\nTotal evidence items: {len(corpus)}")

    # Select one example of each content type
    selected = {}

    for item in corpus:

        content_type = item["content_type"]

        if content_type not in selected:
            selected[content_type] = item

    print("\nSelected examples:")

    for content_type, item in selected.items():

        print("\n" + "-" * 60)
        print(f"Content type : {content_type}")
        print(f"Evidence ID  : {item['evidence_id']}")
        print(f"Paper ID     : {item['paper_id']}")
        print(f"Figure type  : {item['figure_type']}")
        print(f"Caption      : {item['caption']}")
        print(f"Image path   : {item['image_path']}")

    # Check that all selected images exist
    print("\n" + "=" * 60)
    print("IMAGE EXISTENCE CHECK")
    print("=" * 60)

    for content_type, item in selected.items():

        exists = os.path.exists(item["image_path"])

        print(
            f"{content_type:10s} -> "
            f"{'FOUND' if exists else 'MISSING'}"
        )


if __name__ == "__main__":
    main()