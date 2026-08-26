from huggingface_hub import hf_hub_download

file_path = hf_hub_download(
    repo_id="google/spiqa",
    filename="test-A/SPIQA_testA_Images.zip",
    repo_type="dataset",
    local_dir="data"
)

print("Downloaded to:", file_path)