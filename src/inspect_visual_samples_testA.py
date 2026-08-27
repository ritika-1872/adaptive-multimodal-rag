import json
import os
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt


CORPUS_PATH = "data/processed/evidence_corpus.json"


def main():

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # Select one representative image for each figure type
    selected = OrderedDict()

    for item in corpus:

        figure_type = item["figure_type"]

        if figure_type not in selected:
            selected[figure_type] = item

    print("Selected visual categories:")

    for figure_type, item in selected.items():
        print(
            f"{figure_type:15s} -> "
            f"{item['evidence_id']}"
        )

    # Limit to first 10 categories
    samples = list(selected.values())[:10]

    # Create figure
    fig, axes = plt.subplots(
        2,
        5,
        figsize=(20, 8)
    )

    axes = axes.flatten()

    for i, item in enumerate(samples):

        image_path = item["image_path"]

        try:
            image = Image.open(image_path)

            axes[i].imshow(image)
            axes[i].axis("off")

            axes[i].set_title(
                f"{item['figure_type']}\n"
                f"{item['evidence_id']}",
                fontsize=10
            )

        except Exception as e:

            print(
                f"Could not open {image_path}: {e}"
            )

            axes[i].axis("off")

    # Hide unused plots
    for i in range(len(samples), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    output_path = "results/visual_samples.png"

    os.makedirs("results", exist_ok=True)

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print("\nVisual sample saved to:")
    print(output_path)

    plt.show()


if __name__ == "__main__":
    main()