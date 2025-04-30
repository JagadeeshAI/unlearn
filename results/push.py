from huggingface_hub import create_repo, upload_folder
import os

# Set your Hugging Face repo details here
repo_id = "Jagadeesh9580/unlearn"  # 👈 update this if needed
local_folder = "./results"          # Local folder to push
repo_type = "model"                 # It's a model repo

# Check if folder exists
if not os.path.exists(local_folder):
    raise ValueError(f"Local folder '{local_folder}' does not exist!")


# 2. Upload the folder
upload_folder(
    folder_path=local_folder,
    path_in_repo="seg",        # Inside Hugging Face repo -> under /results
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message="Uploading results 🚀",
)

print(f"✅ Successfully pushed '{local_folder}' to Hugging Face repo '{repo_id}'!")