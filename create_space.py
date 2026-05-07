from huggingface_hub import HfApi, login

# Login using your token
login(token="<YOUR_HUGGING_FACE_TOKEN>")

api = HfApi()

repo_id = "SezarTheGreat/mall-customer-segmentation-app"

print(f"Creating Space {repo_id}...")
# Create a Gradio Space
api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)

print("Uploading files to the space...")
# Upload necessary files
api.upload_file(path_or_fileobj="app.py", path_in_repo="app.py", repo_id=repo_id, repo_type="space")
api.upload_file(path_or_fileobj="requirements.txt", path_in_repo="requirements.txt", repo_id=repo_id, repo_type="space")
api.upload_file(path_or_fileobj="kmeans_model.pkl", path_in_repo="kmeans_model.pkl", repo_id=repo_id, repo_type="space")
api.upload_file(path_or_fileobj="Mall_Customers.csv", path_in_repo="Mall_Customers.csv", repo_id=repo_id, repo_type="space")

print(f"Successfully deployed! View your app at: https://huggingface.co/spaces/{repo_id}")
