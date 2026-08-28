import json
import os

import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATASET_PATH = "data/processed/spiqa_testA_processed.json"

METADATA_PATH = "results/image_index/image_metadata.json"

EMBEDDINGS_PATH = "results/image_index/image_embeddings.pt"

RESULTS_PATH = "results/retrieval_evaluation.json"


# --------------------------------------------------
# Model
# --------------------------------------------------

MODEL_NAME = "google/siglip2-base-patch16-224"


def main():

    print("=" * 60)
    print("FULL SPIQA TEST-A RETRIEVAL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Device
    # --------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDevice: {device}")

    # --------------------------------------------------
    # 2. Load dataset
    # --------------------------------------------------

    print("\nLoading processed dataset...")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Questions: {len(dataset)}")

    # --------------------------------------------------
    # 3. Load metadata
    # --------------------------------------------------

    print("\nLoading image metadata...")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"Image metadata records: {len(metadata)}")

    # --------------------------------------------------
    # 4. Load image embeddings
    # --------------------------------------------------

    print("\nLoading image embeddings...")

    image_embeddings = torch.load(
        EMBEDDINGS_PATH,
        map_location="cpu"
    )

    # Make sure embeddings are normalized
    image_embeddings = image_embeddings / image_embeddings.norm(
        dim=-1,
        keepdim=True
    )

    print(
        f"Image embedding shape: "
        f"{image_embeddings.shape}"
    )

    # --------------------------------------------------
    # 5. Check consistency
    # --------------------------------------------------

    if len(metadata) != image_embeddings.shape[0]:

        raise ValueError(
            "Number of metadata records does not "
            "match number of image embeddings."
        )

    # --------------------------------------------------
    # 6. Load model
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

    # Move image embeddings to device
    image_embeddings = image_embeddings.to(device)

    # --------------------------------------------------
    # 7. Create evidence lookup
    # --------------------------------------------------

    evidence_to_index = {}

    for index, item in enumerate(metadata):

        evidence_to_index[
            item["evidence_id"]
        ] = index

    # --------------------------------------------------
    # 8. Evaluation counters
    # --------------------------------------------------

    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0

    reciprocal_ranks = []

    results = []

    # --------------------------------------------------
    # 9. Evaluate every question
    # --------------------------------------------------

    print("\nEvaluating retrieval...")

    for item in tqdm(
        dataset,
        desc="Questions"
    ):

        question = item["question"]

        # --------------------------------------------------
        # Find correct evidence
        # --------------------------------------------------

        reference = item["reference"]

        correct_evidence_id = None

        for metadata_item in metadata:

            if metadata_item["reference"] == reference:

                correct_evidence_id = (
                    metadata_item["evidence_id"]
                )

                break

        if correct_evidence_id is None:

            print(
                f"\nWARNING: Could not find evidence "
                f"for reference: {reference}"
            )

            continue

        correct_index = evidence_to_index[
            correct_evidence_id
        ]

        # --------------------------------------------------
        # Generate question embedding
        # --------------------------------------------------

        text_inputs = processor(
            text=[question],
            padding="max_length",
            truncation=True,
            max_length=64,
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

        text_embedding = text_outputs.pooler_output

        # Normalize
        text_embedding = text_embedding / text_embedding.norm(
            dim=-1,
            keepdim=True
        )

        # --------------------------------------------------
        # Calculate similarities
        # --------------------------------------------------

        similarities = torch.matmul(
            text_embedding,
            image_embeddings.T
        )

        similarities = similarities.squeeze(0)

        # --------------------------------------------------
        # Rank images
        # --------------------------------------------------

        ranked_indices = torch.argsort(
            similarities,
            descending=True
        )

        # --------------------------------------------------
        # Find correct rank
        # --------------------------------------------------

        correct_positions = (
            ranked_indices == correct_index
        ).nonzero(as_tuple=True)[0]

        if len(correct_positions) == 0:

            continue

        correct_rank = (
            correct_positions[0].item() + 1
        )

        # --------------------------------------------------
        # Recall@K
        # --------------------------------------------------

        if correct_rank <= 1:
            recall_at_1 += 1

        if correct_rank <= 5:
            recall_at_5 += 1

        if correct_rank <= 10:
            recall_at_10 += 1

        # --------------------------------------------------
        # Reciprocal rank
        # --------------------------------------------------

        reciprocal_rank = 1.0 / correct_rank

        reciprocal_ranks.append(
            reciprocal_rank
        )

        # --------------------------------------------------
        # Store result
        # --------------------------------------------------

        top_10 = []

        for rank, index in enumerate(
            ranked_indices[:10],
            start=1
        ):

            index = index.item()

            top_10.append(
                {
                    "rank": rank,
                    "evidence_id": metadata[index][
                        "evidence_id"
                    ],
                    "similarity": float(
                        similarities[index].item()
                    )
                }
            )

        results.append(
            {
                "question": question,
                "reference": reference,
                "correct_evidence_id":
                    correct_evidence_id,
                "correct_rank": correct_rank,
                "top_10": top_10
            }
        )

    # --------------------------------------------------
    # 10. Calculate final metrics
    # --------------------------------------------------

    num_evaluated = len(reciprocal_ranks)

    if num_evaluated == 0:

        raise ValueError(
            "No questions were successfully evaluated."
        )

    recall_1 = recall_at_1 / num_evaluated

    recall_5 = recall_at_5 / num_evaluated

    recall_10 = recall_at_10 / num_evaluated

    mrr = sum(reciprocal_ranks) / num_evaluated

    # --------------------------------------------------
    # 11. Display results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"\nQueries evaluated: {num_evaluated}"
    )

    print(
        f"Candidate images: {len(metadata)}"
    )

    print(
        f"\nRecall@1:  {recall_1:.4f} "
        f"({recall_1 * 100:.2f}%)"
    )

    print(
        f"Recall@5:  {recall_5:.4f} "
        f"({recall_5 * 100:.2f}%)"
    )

    print(
        f"Recall@10: {recall_10:.4f} "
        f"({recall_10 * 100:.2f}%)"
    )

    print(
        f"MRR:       {mrr:.4f}"
    )

    # --------------------------------------------------
    # 12. Save results
    # --------------------------------------------------

    output = {
        "model": MODEL_NAME,
        "num_queries": num_evaluated,
        "num_candidates": len(metadata),
        "recall_at_1": recall_1,
        "recall_at_5": recall_5,
        "recall_at_10": recall_10,
        "mrr": mrr,
        "query_results": results
    }

    os.makedirs(
        os.path.dirname(RESULTS_PATH),
        exist_ok=True
    )

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nDetailed results saved to:"
        f"\n{RESULTS_PATH}"
    )

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()