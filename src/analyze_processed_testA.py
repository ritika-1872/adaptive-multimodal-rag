import json
import os
from collections import Counter


DATA_PATH = "data/processed/spiqa_testA_processed.json"


def main():

    # Load dataset
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 60)
    print("SPIQA TEST-A DATASET ANALYSIS")
    print("=" * 60)

    print(f"\nTotal samples: {len(data)}")

    # --------------------------------------------------
    # 1. Content type distribution
    # --------------------------------------------------

    content_types = Counter(
        item.get("content_type", "missing")
        for item in data
    )

    print("\nContent type distribution:")
    for content_type, count in content_types.items():
        print(f"  {content_type}: {count}")

    # --------------------------------------------------
    # 2. Figure type distribution
    # --------------------------------------------------

    figure_types = Counter(
        item.get("figure_type", "missing")
        for item in data
    )

    print("\nFigure type distribution:")
    for figure_type, count in figure_types.items():
        print(f"  {figure_type}: {count}")

    # --------------------------------------------------
    # 3. Image path analysis
    # --------------------------------------------------

    total_images = 0
    existing_images = 0
    missing_images = 0

    missing_paths = []

    for item in data:

        image_path = item.get("image_path")

        if image_path:
            total_images += 1

            if os.path.exists(image_path):
                existing_images += 1
            else:
                missing_images += 1
                missing_paths.append(image_path)

    print("\nImage path analysis:")
    print(f"  Samples with image path: {total_images}")
    print(f"  Existing image files:    {existing_images}")
    print(f"  Missing image files:     {missing_images}")

    # --------------------------------------------------
    # 4. Reference analysis
    # --------------------------------------------------

    references = [
        item.get("reference")
        for item in data
        if item.get("reference")
    ]

    print("\nReference information:")
    print(f"  Samples with reference: {len(references)}")
    print(f"  Unique references:      {len(set(references))}")

    # --------------------------------------------------
    # 5. Paper analysis
    # --------------------------------------------------

    paper_ids = [
        item.get("paper_id")
        for item in data
        if item.get("paper_id")
    ]

    print("\nPaper information:")
    print(f"  Unique papers: {len(set(paper_ids))}")

    # --------------------------------------------------
    # 6. Check missing important fields
    # --------------------------------------------------

    fields = [
        "paper_id",
        "question",
        "answer",
        "explanation",
        "reference",
        "image_path",
        "caption",
        "content_type",
        "figure_type"
    ]

    print("\nMissing field analysis:")

    for field in fields:

        missing = sum(
            1 for item in data
            if not item.get(field)
        )

        print(f"  {field}: {missing} missing")

    # --------------------------------------------------
    # 7. Duplicate image paths
    # --------------------------------------------------

    image_paths = [
        item.get("image_path")
        for item in data
        if item.get("image_path")
    ]

    image_counts = Counter(image_paths)

    duplicate_images = {
        path: count
        for path, count in image_counts.items()
        if count > 1
    }

    print("\nDuplicate image analysis:")
    print(f"  Unique image paths: {len(image_counts)}")
    print(f"  Images used more than once: {len(duplicate_images)}")

    # --------------------------------------------------
    # 8. Show examples from each content type
    # --------------------------------------------------

    print("\nExamples:")
    
    shown_types = set()

    for item in data:

        content_type = item.get("content_type", "missing")

        if content_type not in shown_types:

            print("\n" + "-" * 60)
            print(f"Content type: {content_type}")
            print(f"Question: {item.get('question')}")
            print(f"Image: {item.get('image_path')}")
            print(f"Caption: {item.get('caption')}")
            print(f"Figure type: {item.get('figure_type')}")

            shown_types.add(content_type)

    # --------------------------------------------------
    # 9. Show missing image examples
    # --------------------------------------------------

    if missing_paths:

        print("\n" + "=" * 60)
        print("FIRST 10 MISSING IMAGE PATHS")
        print("=" * 60)

        for path in missing_paths[:10]:
            print(path)


if __name__ == "__main__":
    main()