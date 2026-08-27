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
    print("QUESTION -> IMAGE RETRIEVAL TEST")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Select device
    # --------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDevice: {device}")

    # --------------------------------------------------
    # 2. Load model
    # --------------------------------------------------

    print("\nLoading model...")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    model = AutoModel.from_pretrained(MODEL_NAME)

    model = model.to(device)
    model.eval()

    print("Model loaded successfully.")

    # --------------------------------------------------
    # 3. Load evidence corpus
    # --------------------------------------------------

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    print(f"\nTotal evidence items available: {len(corpus)}")

    # --------------------------------------------------
    # 4. Select test query
    # --------------------------------------------------

    question = (
        "What is the role of the knowledge gates "
        "in the KEHNN architecture?"
    )

    correct_evidence_id = "evidence_0003"

    print("\nQuery:")
    print(question)

    print(f"\nCorrect evidence: {correct_evidence_id}")

    # --------------------------------------------------
    # 5. Select 10 candidate images
    # --------------------------------------------------

    candidates = corpus[:10]

    print("\nCandidate evidence:")
    
    for item in candidates:
        print(
            f"  {item['evidence_id']} "
            f"({item['content_type']}, "
            f"{item['figure_type']})"
        )

    # --------------------------------------------------
    # 6. Prepare question
    # --------------------------------------------------

    text_inputs = processor(
        text=[question],
        padding="max_length",
        return_tensors="pt"
    )

    text_inputs = {
        key: value.to(device)
        for key, value in text_inputs.items()
    }

    # --------------------------------------------------
    # 7. Generate question embedding
    # --------------------------------------------------

    print("\nGenerating question embedding...")

    with torch.no_grad():

        text_outputs = model.get_text_features(
            **text_inputs
        )

    # Extract embedding tensor
    text_embedding = text_outputs.pooler_output

    # Normalize
    text_embedding = text_embedding / text_embedding.norm(
        dim=-1,
        keepdim=True
    )

    # --------------------------------------------------
    # 8. Load candidate images
    # --------------------------------------------------

    images = []

    for item in candidates:

        image = Image.open(
            item["image_path"]
        ).convert("RGB")

        images.append(image)

    print("All candidate images loaded successfully.")

    # --------------------------------------------------
    # 9. Generate image embeddings
    # --------------------------------------------------

    print("\nGenerating image embeddings...")

    image_inputs = processor(
        images=images,
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

    # Extract embedding tensor
    image_embeddings = image_outputs.pooler_output

    # Normalize
    image_embeddings = image_embeddings / image_embeddings.norm(
        dim=-1,
        keepdim=True
    )

    # --------------------------------------------------
    # 10. Calculate similarities
    # --------------------------------------------------

    similarities = torch.matmul(
        text_embedding,
        image_embeddings.T
    )

    similarities = similarities.squeeze(0)

    # --------------------------------------------------
    # 11. Sort by similarity
    # --------------------------------------------------

    ranked_indices = torch.argsort(
        similarities,
        descending=True
    )

    # --------------------------------------------------
    # 12. Display results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("RETRIEVAL RESULTS")
    print("=" * 60)

    print(
        f"\n{'Rank':<8}"
        f"{'Evidence ID':<18}"
        f"{'Similarity':<15}"
        f"{'Correct?'}"
    )

    print("-" * 60)

    for rank, index in enumerate(ranked_indices, start=1):

        index = index.item()

        item = candidates[index]

        evidence_id = item["evidence_id"]

        similarity = similarities[index].item()

        correct = (
            "YES"
            if evidence_id == correct_evidence_id
            else "NO"
        )

        print(
            f"{rank:<8}"
            f"{evidence_id:<18}"
            f"{similarity:<15.4f}"
            f"{correct}"
        )

    # --------------------------------------------------
    # 13. Calculate Rank of correct evidence
    # --------------------------------------------------

    correct_rank = None

    for rank, index in enumerate(
        ranked_indices,
        start=1
    ):

        index = index.item()

        if candidates[index]["evidence_id"] == correct_evidence_id:

            correct_rank = rank
            break

    print("\n" + "=" * 60)

    if correct_rank is not None:

        print(
            f"Correct evidence rank: {correct_rank}"
        )

        print(
            f"Recall@1: "
            f"{'1' if correct_rank <= 1 else '0'}"
        )

        print(
            f"Recall@5: "
            f"{'1' if correct_rank <= 5 else '0'}"
        )

        print(
            f"Recall@10: "
            f"{'1' if correct_rank <= 10 else '0'}"
        )

    else:

        print("Correct evidence was not found.")

    print("=" * 60)


if __name__ == "__main__":
    main()