from huggingface_hub import list_repo_files, hf_hub_download
import os

repo_id = "Jagadeesh9580/unlearn"
repo_type = "model"
local_dir = "./results"

# Create local directory
os.makedirs(local_dir, exist_ok=True)

# List all files in the repo
all_files = list_repo_files(repo_id=repo_id, repo_type=repo_type)

# Filter for files inside 'results/' or 'seg/'
target_folders = ['results/', 'seg/']
files_to_download = [f for f in all_files if any(f.startswith(folder) for folder in target_folders)]

# Download each file
for file_path in files_to_download:
    local_path = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=file_path,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print(f"✅ Downloaded: {local_path}")
