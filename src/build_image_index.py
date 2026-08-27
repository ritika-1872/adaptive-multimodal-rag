import json
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel


# --------------------------------------------------
# Paths
# --------------------------------------------------

CORPUS_PATH = "data/processed/evidence_corpus.json"

OUTPUT_DIR = "results/image_index"

EMBEDDINGS_PATH = os.path.join(
    OUTPUT_DIR,
    "image_embeddings.pt"
)

METADATA_PATH = os.path.join(
    OUTPUT_DIR,
    "image_metadata.json"
)


# --------------------------------------------------
# Model
# --------------------------------------------------

MODEL_NAME = "google/siglip2-base-patch16-224"

# CPU-friendly batch size
BATCH_SIZE = 4


def main():

    print("=" * 60)
    print("BUILDING IMAGE EMBEDDING INDEX")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Select device
    # --------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDevice: {device}")
    print(f"Batch size: {BATCH_SIZE}")

    # --------------------------------------------------
    # 2. Load evidence corpus
    # --------------------------------------------------

    print("\nLoading evidence corpus...")

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    print(f"Evidence items: {len(corpus)}")

    # --------------------------------------------------
    # 3. Load model
    # --------------------------------------------------

    print("\nLoading SigLIP2 model...")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME
    )

    model = model.to(device)
    model.eval()

    print("Model loaded successfully.")

    # --------------------------------------------------
    # 4. Process images in batches
    # --------------------------------------------------

    all_embeddings = []
    valid_metadata = []

    print("\nGenerating image embeddings...")

    for start in tqdm(
        range(0, len(corpus), BATCH_SIZE),
        desc="Processing images"
    ):

        batch = corpus[
            start:start + BATCH_SIZE
        ]

        images = []
        batch_metadata = []

        # ------------------------------------------
        # Load images
        # ------------------------------------------

        for item in batch:

            image_path = item["image_path"]

            try:

                image = Image.open(
                    image_path
                ).convert("RGB")

                images.append(image)
                batch_metadata.append(item)

            except Exception as e:

                print(
                    f"\nWARNING: Could not load image:"
                    f"\n{image_path}"
                    f"\nError: {e}"
                )

        # ------------------------------------------
        # Skip empty batch
        # ------------------------------------------

        if not images:
            continue

        # ------------------------------------------
        # Prepare images
        # ------------------------------------------

        image_inputs = processor(
            images=images,
            return_tensors="pt"
        )

        image_inputs = {
            key: value.to(device)
            for key, value in image_inputs.items()
        }

        # ------------------------------------------
        # Generate embeddings
        # ------------------------------------------

        with torch.no_grad():

            image_outputs = model.get_image_features(
                **image_inputs
            )

        # Extract pooled embeddings
        embeddings = image_outputs.pooler_output

        # ------------------------------------------
        # Normalize embeddings
        # ------------------------------------------

        embeddings = embeddings / embeddings.norm(
            dim=-1,
            keepdim=True
        )

        # Move to CPU
        embeddings = embeddings.cpu()

        all_embeddings.append(embeddings)
        valid_metadata.extend(batch_metadata)

    # --------------------------------------------------
    # 5. Combine embeddings
    # --------------------------------------------------

    print("\nCombining embeddings...")

    if not all_embeddings:
        print("ERROR: No embeddings were generated.")
        return

    image_embeddings = torch.cat(
        all_embeddings,
        dim=0
    )

    print(
        f"Embedding tensor shape: "
        f"{image_embeddings.shape}"
    )

    # --------------------------------------------------
    # 6. Create output directory
    # --------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------
    # 7. Save embeddings
    # --------------------------------------------------

    torch.save(
        image_embeddings,
        EMBEDDINGS_PATH
    )

    # --------------------------------------------------
    # 8. Save metadata
    # --------------------------------------------------

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            valid_metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------
    # 9. Final verification
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("IMAGE INDEX CREATED")
    print("=" * 60)

    print(
        f"\nImages indexed: "
        f"{len(valid_metadata)}"
    )

    print(
        f"Embedding dimension: "
        f"{image_embeddings.shape[1]}"
    )

    print(
        f"\nEmbeddings saved to:"
        f"\n{EMBEDDINGS_PATH}"
    )

    print(
        f"\nMetadata saved to:"
        f"\n{METADATA_PATH}"
    )

    # --------------------------------------------------
    # 10. Verify counts
    # --------------------------------------------------

    if len(valid_metadata) == len(corpus):

        print(
            "\nSUCCESS: All evidence images "
            "were indexed."
        )

    else:

        print(
            f"\nWARNING: Expected {len(corpus)} "
            f"images but indexed "
            f"{len(valid_metadata)}."
        )


if __name__ == "__main__":
    main()