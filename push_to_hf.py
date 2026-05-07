from huggingface_hub import HfApi, login

# Login with the provided token
login(token="<YOUR_HUGGING_FACE_TOKEN>")

# Initialize the API
api = HfApi()

repo_id = "SezarTheGreat/mall-customer-segmentation-kmeans"

print(f"Creating repository {repo_id}...")
# Create the repository on Hugging Face (does nothing if it already exists)
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

print("Uploading kmeans_model.pkl...")
# Upload the model file
api.upload_file(
    path_or_fileobj="kmeans_model.pkl",
    path_in_repo="kmeans_model.pkl",
    repo_id=repo_id,
    repo_type="model",
)

print("Uploading Mall_Customers.csv...")
# Upload the dataset
api.upload_file(
    path_or_fileobj="Mall_Customers.csv",
    path_in_repo="Mall_Customers.csv",
    repo_id=repo_id,
    repo_type="model",
)

print("Uploading README.md...")
# Upload the README for the model card
api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="model",
)

print(f"Success! Model uploaded to: https://huggingface.co/{repo_id}")
