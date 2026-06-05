from huggingface_hub import HfApi, login
import os

# Login using your token
login(token="<YOUR_HUGGING_FACE_TOKEN>")

api = HfApi()

repo_id = "SezarTheGreat/mall-customer-segmentation-app"

print(f"Creating Space {repo_id}...")
# Create a Gradio Space
api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)

print("Uploading files to the space...")
# List of all files necessary to run the state-of-the-art segmentation Gradio app
files_to_upload = [
    "app.py",
    "requirements.txt",
    "supervised_explainability.py",
    "feature_names.json",
    "clv_bgf_params.pkl",
    "clv_ggf_params.pkl",
    "pca_pipeline.pkl",
    "gmm_model.pkl",
    "xgb_classifier.pkl",
    "baseline_training_data.parquet"
]

for file in files_to_upload:
    if os.path.exists(file):
        print(f"Uploading {file}...")
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=file,
            repo_id=repo_id,
            repo_type="space"
        )
    else:
        print(f"Warning: {file} not found locally. Skipping.")

print(f"\nSuccessfully deployed! View your app at: https://huggingface.co/spaces/{repo_id}")
