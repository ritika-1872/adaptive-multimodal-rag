import json
from pathlib import Path


# Paths
JSON_PATH = Path("data/test-A/SPIQA_testA.json")
IMAGE_DIR = Path("data/test-A/SPIQA_testA_Images/SPIQA_testA_Images")
OUTPUT_PATH = Path("data/processed/spiqa_testA_processed.json")


# Load the SPIQA JSON
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


processed_data = []


# Go through every paper
for paper_id, paper_data in data.items():

    all_figures = paper_data.get("all_figures", {})
    qa_list = paper_data.get("qa", [])

    # Go through every question in the paper
    for qa in qa_list:

        question = qa.get("question")
        answer = qa.get("answer")
        explanation = qa.get("explanation")
        reference = qa.get("reference")

        # Get information about the referenced figure/table
        figure_info = all_figures.get(reference, {})

        caption = figure_info.get("caption")
        content_type = figure_info.get("content_type")
        figure_type = figure_info.get("figure_type")

        # Path to the actual image
        image_path = IMAGE_DIR / paper_id / reference

        processed_item = {
            "paper_id": paper_id,
            "question": question,
            "answer": answer,
            "explanation": explanation,
            "reference": reference,
            "image_path": str(image_path),
            "caption": caption,
            "content_type": content_type,
            "figure_type": figure_type
        }

        processed_data.append(processed_item)


# Create processed-data directory if necessary
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# Save the processed dataset
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, indent=2, ensure_ascii=False)


print("Dataset preparation complete!")
print("Total QA pairs:", len(processed_data))
print("Saved to:", OUTPUT_PATH)