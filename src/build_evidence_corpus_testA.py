import json
import os


INPUT_PATH = "data/processed/spiqa_testA_processed.json"
OUTPUT_PATH = "data/processed/evidence_corpus.json"


def normalize_figure_type(figure_type):
    """
    Normalize inconsistent figure type labels.
    """

    if not figure_type:
        return "unknown"

    value = figure_type.strip().lower()

    # Remove common markdown-style prefixes
    value = value.replace("**", "").strip()

    # Remove unnecessary punctuation
    value = value.rstrip(".")

    # Normalize common variations
    if value == "plot":
        return "plot"

    if value == "schematic":
        return "schematic"

    if value == "table":
        return "table"

    if value == "map":
        return "map"

    if value == "photograph(s)":
        return "photograph"

    if value == "photographs":
        return "photograph"

    if value == "other":
        return "other"

    if value == "n/a":
        return "unknown"

    if value == "":
        return "unknown"

    return value


def main():

    # -----------------------------------------
    # Load processed dataset
    # -----------------------------------------

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples.")

    evidence_corpus = []

    # -----------------------------------------
    # Create evidence records
    # -----------------------------------------

    for index, item in enumerate(data):

        image_path = item["image_path"]

        # Verify image exists
        if not os.path.exists(image_path):
            print(f"WARNING: Image not found: {image_path}")
            continue

        evidence = {
            "evidence_id": f"evidence_{index:04d}",
            "paper_id": item["paper_id"],
            "reference": item["reference"],
            "image_path": image_path,
            "caption": item["caption"],
            "content_type": item["content_type"],
            "figure_type": normalize_figure_type(
                item.get("figure_type")
            )
        }

        evidence_corpus.append(evidence)

    # -----------------------------------------
    # Save corpus
    # -----------------------------------------

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            evidence_corpus,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\nEvidence corpus created successfully.")

    print(f"Total evidence items: {len(evidence_corpus)}")
    print(f"Output file: {OUTPUT_PATH}")

    # -----------------------------------------
    # Show first example
    # -----------------------------------------

    if evidence_corpus:

        print("\nFirst evidence item:")
        print(
            json.dumps(
                evidence_corpus[0],
                indent=2,
                ensure_ascii=False
            )
        )


if __name__ == "__main__":
    main()