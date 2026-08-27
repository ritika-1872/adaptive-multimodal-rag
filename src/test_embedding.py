import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel


# --------------------------------------------------
# Paths
# --------------------------------------------------

CORPUS_PATH = "data/processed/evidence_corpus.json"


# --------------------------------------------------
# Model
# --------------------------------------------------

MODEL_NAME = "google/siglip2-base-patch16-224"


def main():

    print("=" * 60)
    print("MULTIMODAL EMBEDDING TEST")
    print("=" * 60)

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDevice: {device}")

    # --------------------------------------------------
    # Load model and processor
    # --------------------------------------------------

    print("\nLoading model...")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    model = AutoModel.from_pretrained(MODEL_NAME)

    model = model.to(device)
    model.eval()

    print("Model loaded successfully.")

    # --------------------------------------------------
    # Load evidence corpus
    # --------------------------------------------------

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    print(f"\nEvidence items: {len(corpus)}")

    # --------------------------------------------------
    # Select evidence_0003
    # --------------------------------------------------

    evidence = next(
        item for item in corpus
        if item["evidence_id"] == "evidence_0003"
    )

    question = (
        "What is the role of the knowledge gates "
        "in the KEHNN architecture?"
    )

    image_path = evidence["image_path"]

    print("\nQuestion:")
    print(question)

    print("\nImage:")
    print(image_path)

    # --------------------------------------------------
    # Load image
    # --------------------------------------------------

    image = Image.open(image_path).convert("RGB")

    print("\nImage loaded successfully.")
    print("Image size:", image.size)

    # --------------------------------------------------
    # Encode text
    # --------------------------------------------------

    print("\nGenerating text embedding...")

    text_inputs = processor(
        text=[question],
        padding="max_length",
        return_tensors="pt"
    )

    text_inputs = {
        key: value.to(device)
        for key, value in text_inputs.items()
    }

    with torch.no_grad():

        text_outputs = model.get_text_features(
            **text_inputs
        )
    # Extract the actual embedding tensor
    text_embedding = text_outputs.pooler_output

    # Normalize
    text_embedding = text_embedding / text_embedding.norm(
        dim=-1,
        keepdim=True
    )

    # --------------------------------------------------
    # Encode image
    # --------------------------------------------------

    print("Generating image embedding...")

    image_inputs = processor(
        images=image,
        return_tensors="pt"
    )

    image_inputs = {
        key: value.to(device)
        for key, value in image_inputs.items()
    }

    with torch.no_grad():

        image_outputs = model.get_image_features(
            **image_inputs
        )
    # Extract the actual embedding tensor
    image_embedding = image_outputs.pooler_output
    # Normalize
    image_embedding = image_embedding / image_embedding.norm(
        dim=-1,
        keepdim=True
    )

    # --------------------------------------------------
    # Similarity
    # --------------------------------------------------

    similarity = torch.matmul(
        text_embedding,
        image_embedding.T
    )

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    print("\nText embedding shape:")
    print(text_embedding.shape)

    print("\nImage embedding shape:")
    print(image_embedding.shape)

    print("\nCosine similarity:")
    print(similarity.item())

    print("\nEmbedding test completed successfully.")


if __name__ == "__main__":
    main()